#!/usr/bin/env python
"""
LatentMAS-RP Training Entry Point

Single-model multi-role training: one backbone LLM plays all four agent roles
(Planner, Critic, Refiner, Judger) via different system prompts.  Communication
between roles happens through the accumulated KV cache.  Only the Judger's
output is evaluated against the task reward; the backbone is trained end-to-end
via PPO/GiGPO on the Judger's trajectories.

The optional Reading Policy (GateNet) is a lightweight add-on trained separately
via train_reading_policy.py and is not part of this backbone training loop.
"""

import os
import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="ppo_trainer_latent_mas", version_base=None)
def main(config):
    run_training(config)


def run_training(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                }
            },
            num_cpus=config.ray_init.num_cpus,
        )

    runner = LatentMASTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class LatentMASTaskRunner:
    """Ray actor that owns the full training lifecycle."""

    def run(self, config):
        from pprint import pprint
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
        from verl.single_controller.ray import RayWorkerGroup
        from agent_system.multi_turn_rollout.rollout_loop import MultiAgentTrajectoryCollector

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # ------------------------------------------------------------------ #
        # Model / tokenizer                                                    #
        # ------------------------------------------------------------------ #
        model_path = config.actor_rollout_ref.model.path
        local_path = copy_to_local(
            model_path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # ------------------------------------------------------------------ #
        # Datasets                                                             #
        # ------------------------------------------------------------------ #
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset   = create_rl_dataset(config.data.val_files,   config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # ------------------------------------------------------------------ #
        # Reward functions                                                     #
        # ------------------------------------------------------------------ #
        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == "episode":
            from agent_system.reward_manager.episode import EpisodeRewardManager
            reward_manager_cls = EpisodeRewardManager
        else:
            raise NotImplementedError(f"Unknown reward_manager: {reward_manager_name}")

        reward_fn     = reward_manager_cls(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)

        # ------------------------------------------------------------------ #
        # Trajectory collector                                                 #
        # ------------------------------------------------------------------ #
        traj_collector = MultiAgentTrajectoryCollector(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            reward_fn=reward_fn,
        )

        # ------------------------------------------------------------------ #
        # Worker classes                                                       #
        # ------------------------------------------------------------------ #
        strategy = config.actor_rollout_ref.actor.strategy
        if strategy in ("fsdp", "fsdp2"):
            assert config.critic.strategy in ("fsdp", "fsdp2")
            from verl.workers.fsdp_workers import (
                ActorRolloutRefWorker,
                AsyncActorRolloutRefWorker,
                CriticWorker,
            )
            actor_rollout_cls   = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif strategy == "megatron":
            assert config.critic.strategy == "megatron"
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            actor_rollout_cls    = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError(f"Unknown actor strategy: {strategy}")

        # ------------------------------------------------------------------ #
        # Resource pool                                                        #
        # ------------------------------------------------------------------ #
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic:       ray.remote(CriticWorker),
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic:       global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ("fsdp", "fsdp2"):
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError(f"Unknown reward_model strategy: {config.reward_model.strategy}")
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping,
        )

        # ------------------------------------------------------------------ #
        # Trainer                                                              #
        # ------------------------------------------------------------------ #
        trainer = RayPPOTrainer(
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
            envs=None,
            val_envs=None,
        )

        trainer.init_workers()
        trainer.fit()


# --------------------------------------------------------------------------- #
# Dataset / sampler helpers                                                    #
# --------------------------------------------------------------------------- #

def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError("Custom dataset class must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset

    return dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )


def create_rl_sampler(data_config, dataset):
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.shuffle:
        generator = torch.Generator()
        generator.manual_seed(data_config.get("seed", 1))
        return RandomSampler(data_source=dataset, generator=generator)
    return SequentialSampler(data_source=dataset)


if __name__ == "__main__":
    main()
