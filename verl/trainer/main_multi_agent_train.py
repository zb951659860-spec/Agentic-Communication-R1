#!/usr/bin/env python
"""
Multi-Agent Sequential Training with Single Agent Updates
基于main_ppo.py修改，支持多个agent轮流训练
"""

import os
import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer_latent_mas", version_base=None)
def main(config):
    run_multi_agent_training(config)


def run_multi_agent_training(config) -> None:
    """
    执行多agent轮流训练
    每个epoch训练一个agent，其他agents作为固定协作者
    """
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    runner = MultiAgentTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class MultiAgentTaskRunner:
    """
    Multi-agent训练任务运行器
    管理多个agent的轮流训练
    """
    
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # 获取multi-agent配置
        n_agents = config.env.get('n_agents', 1)
        agent_models = config.multi_agent.get('agent_models', [])
        training_strategy = config.multi_agent.get('training_strategy', 'sequential')  # sequential or simultaneous
        
        assert len(agent_models) == n_agents, \
            f"Number of agent models ({len(agent_models)}) must match n_agents ({n_agents})"
        
        print(f"=" * 60)
        print(f"Multi-Agent Training Configuration:")
        print(f"  Number of agents: {n_agents}")
        print(f"  Training strategy: {training_strategy}")
        print(f"  Agent models:")
        for i, model in enumerate(agent_models):
            print(f"    Agent {i}: {model}")
        print(f"=" * 60)
        
        # 初始化tokenizer和processor（所有agent共享）
        from verl.utils import hf_processor, hf_tokenizer
        
        # 使用第一个agent的模型初始化tokenizer（假设所有agent使用相同的tokenizer）
        base_model_path = agent_models[0]
        local_path = copy_to_local(base_model_path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        
        # 创建数据集
        from verl.utils.dataset.rl_dataset import collate_fn
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        
        # 创建奖励函数
        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            from agent_system.reward_manager.episode import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError
        
        reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
        
        # 初始化所有agent的WorkerGroups（用于固定agents）
        fixed_agent_wgs = []
        for agent_idx in range(n_agents):
            if agent_idx > 0:  # 第一个agent会作为训练agent单独创建
                agent_config = self._create_agent_config(config, agent_models[agent_idx], agent_idx)
                agent_wg = self._create_worker_group(agent_config, is_trainable=False)
                fixed_agent_wgs.append(agent_wg)
        
        # 轮流训练每个agent
        for epoch in range(config.trainer.total_epochs):
            # 确定当前训练的agent
            current_agent_idx = epoch % n_agents
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}: Training Agent {current_agent_idx}")
            print(f"{'='*60}")
            
            # 更新当前agent的配置
            current_agent_config = self._create_agent_config(
                config, 
                agent_models[current_agent_idx], 
                current_agent_idx
            )
            
            # 创建trajectory collector（指定当前训练的agent）
            from agent_system.multi_turn_rollout.rollout_loop import MultiAgentTrajectoryCollector
            
            traj_collector = MultiAgentTrajectoryCollector(
                config=current_agent_config,
                tokenizer=tokenizer,
                processor=processor,
                reward_fn=reward_fn,
                agent_models=agent_models,
                current_agent_idx=current_agent_idx,
            )
            
            # 为当前agent创建trainer
            trainer = self._create_trainer_for_agent(
                config=current_agent_config,
                tokenizer=tokenizer,
                processor=processor,
                traj_collector=traj_collector,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                train_sampler=train_sampler,
                collate_fn=collate_fn,
                fixed_agent_wgs=fixed_agent_wgs,
                current_agent_idx=current_agent_idx,
            )
            
            # 训练当前agent
            trainer.init_workers()
            trainer.fit_single_epoch()  # 训练一个epoch
            
            # 保存当前agent的checkpoint
            self._save_agent_checkpoint(trainer, current_agent_idx, epoch)
            
            # 更新fixed_agent_wgs中对应的agent（如果刚训练的不是第0个）
            if current_agent_idx > 0:
                # 更新fixed_agent_wgs列表中的对应agent
                fixed_agent_wgs[current_agent_idx - 1] = trainer.actor_rollout_wg
            
            # 清理资源
            if epoch < config.trainer.total_epochs - 1:  # 不是最后一个epoch
                trainer.cleanup()
        
        print(f"\n{'='*60}")
        print(f"Multi-Agent Training Complete!")
        print(f"{'='*60}")
    
    def _create_agent_config(self, base_config, agent_model_path, agent_idx):
        """
        为特定agent创建配置
        """
        from omegaconf import OmegaConf
        
        # 深拷贝基础配置
        agent_config = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
        
        # 更新模型路径
        agent_config.actor_rollout_ref.model.path = agent_model_path
        agent_config.critic.model.path = agent_model_path
        
        # 添加agent标识
        agent_config.current_agent_idx = agent_idx
        
        return agent_config
    
    def _create_worker_group(self, config, is_trainable=True):
        """
        创建WorkerGroup（用于固定的agents）
        """
        # 这里简化实现，实际需要根据config创建相应的WorkerGroup
        # 参考main_ppo.py中的实现
        pass
    
    def _create_trainer_for_agent(
        self,
        config,
        tokenizer,
        processor,
        traj_collector,
        reward_fn,
        val_reward_fn,
        train_dataset,
        val_dataset,
        train_sampler,
        collate_fn,
        fixed_agent_wgs,
        current_agent_idx,
    ):
        """
        为特定agent创建trainer
        """
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.single_controller.ray import RayWorkerGroup
        
        # Worker类定义（参考main_ppo.py）
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
            
            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            
            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError
        
        # Role mapping
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }
        
        # Resource pool
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }
        
        # Reward model（如果启用）
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        
        # Reference policy（如果需要）
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id
        
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        
        # 创建修改版的RayPPOTrainer，支持fixed_agent_wgs
        trainer = MultiAgentRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            fixed_agent_wgs=fixed_agent_wgs,
            current_agent_idx=current_agent_idx,
        )
        
        return trainer
    
    def _save_agent_checkpoint(self, trainer, agent_idx, epoch):
        """
        保存特定agent的checkpoint
        """
        checkpoint_dir = os.path.join(
            trainer.config.trainer.default_local_dir,
            f"agent_{agent_idx}",
            f"epoch_{epoch}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存actor
        actor_path = os.path.join(checkpoint_dir, "actor")
        trainer.actor_rollout_wg.save_checkpoint(actor_path, None, epoch)
        
        # 保存critic
        if trainer.use_critic:
            critic_path = os.path.join(checkpoint_dir, "critic")
            trainer.critic_wg.save_checkpoint(critic_path, None, epoch)
        
        print(f"Saved checkpoint for agent {agent_idx} at epoch {epoch}: {checkpoint_dir}")


class MultiAgentRayPPOTrainer:
    """
    修改版的RayPPOTrainer，支持multi-agent训练
    继承或包装原有的RayPPOTrainer
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        processor,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        reward_fn,
        val_reward_fn,
        train_dataset,
        val_dataset,
        collate_fn,
        train_sampler,
        device_name,
        traj_collector,
        fixed_agent_wgs=None,
        current_agent_idx=0,
    ):
        # 导入原始的RayPPOTrainer
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        
        # 创建基础trainer（不传envs）
        self.base_trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
            traj_collector=traj_collector,
            envs=None,  # 不使用环境
            val_envs=None,
        )
        
        # 额外的multi-agent相关属性
        self.fixed_agent_wgs = fixed_agent_wgs
        self.current_agent_idx = current_agent_idx
        
        # 代理基础trainer的属性
        self.config = self.base_trainer.config
        self.actor_rollout_wg = self.base_trainer.actor_rollout_wg
        self.critic_wg = self.base_trainer.critic_wg if hasattr(self.base_trainer, 'critic_wg') else None
        self.use_critic = self.base_trainer.use_critic if hasattr(self.base_trainer, 'use_critic') else False
    
    def init_workers(self):
        """初始化workers"""
        self.base_trainer.init_workers()
    
    def fit_single_epoch(self):
        """训练一个epoch"""
        # 修改traj_collector的multi_turn_loop调用
        # 传入fixed_agent_wgs
        original_multi_turn_loop = self.base_trainer.traj_collector.multi_turn_loop
        
        def wrapped_multi_turn_loop(gen_batch, actor_rollout_wg, envs=None, is_train=True):
            return original_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                fixed_agent_wgs=self.fixed_agent_wgs,
                is_train=is_train,
            )
        
        # 临时替换方法
        self.base_trainer.traj_collector.multi_turn_loop = wrapped_multi_turn_loop
        
        # 调用基础trainer的fit方法（只训练一个epoch）
        # 这里需要修改配置或调用特定方法
        original_epochs = self.config.trainer.total_epochs
        self.config.trainer.total_epochs = 1
        self.base_trainer.fit()
        self.config.trainer.total_epochs = original_epochs
        
        # 恢复原始方法
        self.base_trainer.traj_collector.multi_turn_loop = original_multi_turn_loop
    
    def cleanup(self):
        """清理资源"""
        # 清理ray workers等资源
        if hasattr(self.base_trainer, 'cleanup'):
            self.base_trainer.cleanup()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset (复用main_ppo.py的实现)"""
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset (复用main_ppo.py的实现)"""
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
