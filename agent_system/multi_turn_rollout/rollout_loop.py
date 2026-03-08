"""
Multi-Agent Trajectory Collector with Single Agent Training Mode
支持多agent协作，但每次只训练一个agent的版本
集成LatentMAS潜在空间通信机制
"""

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re
import logging
import argparse

# 相对导入 - 从同级包导入
from agent_system.multi_turn_rollout.models import ModelWrapper, _past_length
from agent_system.multi_turn_rollout.utils import setup_logging, extract_gsm8k_answer, normalize_answer
from agent_system.multi_turn_rollout.prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from agent_system.multi_turn_rollout.methods import Agent, default_agents

# 尝试导入Cache用于KV截断
try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


class MultiAgentTrajectoryCollector:
    """
    Multi-agent Trajectory收集器（单agent训练模式）
    
    关键特性：
    - 支持多个agents参与rollout
    - 每次只返回一个agent的trajectory用于训练
    - 其他agents作为固定的协作者参与
    - 支持LatentMAS潜在空间通信机制
    """

    def __init__(
        self, 
        config, 
        tokenizer: PreTrainedTokenizer, 
        processor=None, 
        reward_fn=None,
    ):
        """
        初始化Multi-agent Trajectory Collector (单模型多角色版本)
        
        Parameters:
            config: 配置对象
            tokenizer: HuggingFace tokenizer
            processor: 可选的多模态processor
            reward_fn: 奖励计算函数
        
        Note:
            单模型通过不同prompt扮演planner/critic/refiner/judger四个角色
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fn = reward_fn

        # Setup logging
        self.log_dir = self.config.get('log_dir', '../../log/GSM8K_integrated_test')
        self.logger = setup_logging(self.log_dir)  

        # Multi-agent配置 - 单模型扮演4个角色（LatentMAS风格）
        self.agents = default_agents()  # [planner, critic, refiner, judger]
        self.n_agents = len(self.agents)  # 固定为4个角色
        # 注意：单模型架构，所有角色由同一model_wrapper通过不同prompt扮演
        
        # Action聚合策略（保留用于非LatentMAS模式）
        self.action_reduction_strategy = config.env.get('action_reduction', 'majority_vote') if hasattr(config, 'env') else 'majority_vote'
        self.action_reducer = self._get_action_reducer(self.action_reduction_strategy)
        
        # 其他配置
        self.max_steps = config.env.get('max_steps', 3) if hasattr(config, 'env') else config.get('max_steps', 3)
        self.enable_summary = config.env.get('enable_agent_summary', True) if hasattr(config, 'env') else True
        self.summary_max_length = config.env.get('summary_max_length', 50) if hasattr(config, 'env') else 50

        # ========== LatentMAS 相关配置 ==========
        self.use_latent_mas = config.get('use_latent_mas', False)
        self.latent_steps = config.get('latent_steps', 3)
        self.temperature = config.get('temperature', 0.7)
        self.top_p = config.get('top_p', 0.95)
        self.judger_max_new_tokens = config.get('judger_max_new_tokens', 4096)
        self.prompt_style = config.get('prompt_style', 'sequential')  # 'sequential' or 'hierarchical'
        self.enable_structured_summary = config.get('enable_structured_summary', True)
        
        # 模型包装器初始化
        self.model_wrapper: Optional[ModelWrapper] = None
        self.latent_args: Optional[argparse.Namespace] = None
        
        if self.use_latent_mas:
            model_name = config.get('model_name', None)
            device = config.get('device', 'cuda')
            use_vllm = config.get('use_vllm', False)
            
            if model_name:
                # 构建args对象供ModelWrapper使用
                self.latent_args = self._build_latent_args(config)
                
                self.model_wrapper = ModelWrapper(
                    model_name=model_name,
                    device=torch.device(device),
                    use_vllm=use_vllm,
                    args=self.latent_args
                )
                self.logger.info(f"[LatentMAS] ModelWrapper initialized with model: {model_name}")
            else:
                self.logger.warning("[LatentMAS] use_latent_mas=True but no model_name provided!")

        self.logger.info(f"[MultiAgentTrajectoryCollector] Initialized with {self.n_agents} roles (single model)")
        self.logger.info(f"[MultiAgentTrajectoryCollector] use_latent_mas={self.use_latent_mas}")

    def _build_latent_args(self, config) -> argparse.Namespace:
        """从config构建LatentMAS所需的args对象"""
        args = argparse.Namespace()
        args.model_name = config.get('model_name', 'Qwen/Qwen3-4B')
        args.device = config.get('device', 'cuda')
        args.device2 = config.get('device2', 'cuda:1')
        args.latent_steps = config.get('latent_steps', 3)
        args.max_new_tokens = config.get('max_new_tokens', 4096)
        args.temperature = config.get('temperature', 0.7)
        args.top_p = config.get('top_p', 0.95)
        args.prompt = config.get('prompt_style', 'sequential')
        args.think = config.get('think', False)
        args.latent_space_realign = config.get('latent_space_realign', False)
        args.use_vllm = config.get('use_vllm', False)
        args.use_second_HF_model = config.get('use_second_HF_model', False)
        args.tensor_parallel_size = config.get('tensor_parallel_size', 1)
        args.gpu_memory_utilization = config.get('gpu_memory_utilization', 0.9)
        args.enable_prefix_caching = config.get('enable_prefix_caching', False)
        args.method = 'latent_mas'
        args.task = config.get('task', 'gsm8k')
        args.text_mas_context_length = config.get('text_mas_context_length', -1)
        return args

    # ========== LatentMAS KV Cache 管理方法 ==========
    
    @staticmethod
    def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
        """截取tensor的最后tokens_to_keep个位置"""
        if tokens_to_keep <= 0:
            return tensor[..., 0:0, :].contiguous()
        keep = min(tokens_to_keep, tensor.shape[-2])
        start = tensor.shape[-2] - keep
        return tensor[..., start:, :].contiguous()

    def _truncate_past(self, past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
        """截断past_key_values到指定长度"""
        if past_kv is None or tokens_to_keep <= 0:
            return None
        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            trimmed_legacy = tuple(
                tuple(self._slice_tensor(t, tokens_to_keep) for t in layer)
                for layer in legacy
            )
            return past_kv.__class__.from_legacy_cache(trimmed_legacy)
        trimmed_layers = []
        for layer in past_kv:
            if isinstance(layer, tuple):
                trimmed_layers.append(tuple(self._slice_tensor(t, tokens_to_keep) for t in layer))
            elif torch.is_tensor(layer):
                trimmed_layers.append(self._slice_tensor(layer, tokens_to_keep))
            else:
                trimmed_layers.append(layer)
        return tuple(trimmed_layers)

    # ========== Prompt 构建与摘要解析 ==========
    
    def _build_agent_prompt_with_summary(
        self, 
        role: str, 
        question: str, 
        summary_context: str = "",
        round_num: int = 0
    ) -> List[Dict]:
        """
        构建agent的prompt消息，支持结构化摘要前缀
        
        与原LatentMAS的区别：
        - 摘要作为系统提示前缀注入
        - 支持多轮讨论中的上下文传递
        """
        # 确保 latent_args 存在
        if self.latent_args is None:
            raise RuntimeError(
                "_build_agent_prompt_with_summary requires latent_args to be initialized. "
                "Ensure use_latent_mas=True and model_name is provided in config."
            )
        
        # 使用原有的prompt构建函数
        if self.prompt_style == "sequential":
            messages = build_agent_message_sequential_latent_mas(
                role=role,
                question=question,
                context="",
                method="latent_mas",
                args=self.latent_args
            )
        else:  # hierarchical
            messages = build_agent_message_hierarchical_latent_mas(
                role=role,
                question=question,
                context="",
                method="latent_mas",
                args=self.latent_args
            )
        
        # 如果有上一轮的摘要，作为前缀注入到system message
        if summary_context and round_num > 0:
            original_system = messages[0]["content"]
            messages[0]["content"] = f"""=== Previous Round Summary (Round {round_num}) ===
{summary_context}
=== End of Summary ===

{original_system}"""
        
        return messages

    def _parse_structured_summary(self, judger_output: str) -> str:
        """
        从judger输出中提取/生成结构化摘要
        
        返回格式:
        [Constraints]: ...
        [Key Conclusions]: ...
        [Unresolved Issues]: ...
        """
        summary_parts = []
        
        # 尝试提取约束条件
        constraints_match = re.search(
            r'\[Constraints?\]:\s*(.+?)(?=\[Key|\[Unresolved|$)', 
            judger_output, re.DOTALL | re.IGNORECASE
        )
        if constraints_match:
            summary_parts.append(f"[Constraints]: {constraints_match.group(1).strip()[:200]}")
        else:
            # 从推理过程中提取关键约束
            summary_parts.append("[Constraints]: Derived from problem analysis")
        
        # 尝试提取关键结论
        conclusions_match = re.search(
            r'\[Key Conclusions?\]:\s*(.+?)(?=\[Unresolved|$)', 
            judger_output, re.DOTALL | re.IGNORECASE
        )
        if conclusions_match:
            summary_parts.append(f"[Key Conclusions]: {conclusions_match.group(1).strip()[:200]}")
        else:
            # 提取boxed答案作为关键结论
            boxed_match = re.search(r'\\boxed\{([^}]*)\}', judger_output)
            if boxed_match:
                summary_parts.append(f"[Key Conclusions]: Current answer candidate: {boxed_match.group(1)}")
            else:
                # 截取输出的关键部分
                summary_parts.append(f"[Key Conclusions]: {judger_output[:150].strip()}...")
        
        # 尝试提取未解决分歧
        issues_match = re.search(
            r'\[Unresolved Issues?\]:\s*(.+?)(?=$)', 
            judger_output, re.DOTALL | re.IGNORECASE
        )
        if issues_match:
            summary_parts.append(f"[Unresolved Issues]: {issues_match.group(1).strip()[:200]}")
        else:
            summary_parts.append("[Unresolved Issues]: To be verified in next round")
        
        return "\n".join(summary_parts)

    def _check_answer_correct(self, question: str, answer: str, gold: Optional[str] = None) -> bool:
        """检查答案是否正确"""
        if gold is None:
            self.logger.warning(f"[_check_answer_correct] gold is None, returning False")
            return False
        
        import re
        
        # ========== 1. 从 gold 中提取答案 ==========
        # GSM8K 格式: "...解题过程...\n#### 数字"
        gold_extracted = None
        
        # 尝试提取 #### 后的数字
        gold_match = re.search(r'####\s*(\S+)', gold)
        if gold_match:
            gold_extracted = gold_match.group(1).strip()
        else:
            # 如果没有 #### 格式，假设整个 gold 就是答案
            gold_extracted = gold.strip()
        
        # ========== 2. 从模型输出中提取答案 ==========
        processed_answer = answer
        
        # 尝试提取 </think> 之后的内容（如果有）
        think_end_match = re.search(r'</think>\s*(.*)', answer, re.DOTALL)
        if think_end_match:
            processed_answer = think_end_match.group(1)
        
        # 提取 \boxed{} 中的答案
        pred = extract_gsm8k_answer(processed_answer)
        
        # 如果从 </think> 后没提取到，尝试从整个答案中提取
        if not pred:
            pred = extract_gsm8k_answer(answer)
        
        # ========== 3. 标准化并比较 ==========
        pred_normalized = normalize_answer(pred) if pred else None
        gold_normalized = normalize_answer(gold_extracted) if gold_extracted else None
        
        # 调试日志
        self.logger.info(f"[Answer Check] gold_raw='{gold[:50]}...' -> gold_extracted='{gold_extracted}' -> gold_normalized='{gold_normalized}'")
        self.logger.info(f"[Answer Check] pred_raw='{pred}' -> pred_normalized='{pred_normalized}'")
        self.logger.info(f"[Answer Check] answer_preview: {answer[:200]}...")
        
        is_correct = pred_normalized == gold_normalized if (pred_normalized and gold_normalized) else False
        self.logger.info(f"[Answer Check] is_correct={is_correct}")
        
        return is_correct

    def _get_action_reducer(self, strategy: str):
        """获取action聚合函数"""
        strategies = {
            'majority_vote': self._majority_vote,
            'weighted_vote': self._weighted_vote,
            'first_agent': self._first_agent_vote,
            # 'current_agent' 已移除 - 单模型架构不需要
        }
        
        if strategy not in strategies:
            raise ValueError(
                f"Unknown action reduction strategy: {strategy}. "
                f"Available: {list(strategies.keys())}"
            )
        
        return strategies[strategy]

    def _majority_vote(self, actions: List[str], logprobs: Optional[List[float]] = None) -> str:
        """多数投票聚合"""
        if not actions:
            return ""
        
        vote_counts = Counter(actions)
        most_common = vote_counts.most_common(1)
        
        if most_common:
            return most_common[0][0]
        return actions[0]

    def _weighted_vote(self, actions: List[str], logprobs: Optional[List[float]] = None) -> str:
        """加权投票（基于logprobs）"""
        if not actions:
            return ""
        
        if logprobs is None:
            return self._majority_vote(actions, None)
        
        action_weights = {}
        for action, logprob in zip(actions, logprobs):
            if action not in action_weights:
                action_weights[action] = 0
            action_weights[action] += np.exp(logprob)
        
        best_action = max(action_weights.items(), key=lambda x: x[1])[0]
        return best_action

    def _first_agent_vote(self, actions: List[str], logprobs: Optional[List[float]] = None) -> str:
        """选择第一个agent的决策"""
        if actions:
            return actions[0]
        return ""

    def generate_agent_actions(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        fixed_agent_wgs: Optional[List] = None,  # 其他固定agent的WorkerGroups
        step: Optional[int] = None,
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        生成所有agents的actions（用于legacy模式）
        
        Parameters:
            gen_batch: 输入batch
            actor_rollout_wg: 当前训练agent的WorkerGroup
            fixed_agent_wgs: 其他固定agent的WorkerGroups（可选）
            
        Returns:
            all_agent_actions: 所有agent的actions
            all_agent_logprobs: 所有agent的logprobs
        """
        batch_size = len(gen_batch)
        all_agent_actions = []
        all_agent_logprobs = []
        
        for agent_id in range(self.n_agents):
            # 所有agents都使用actor_rollout_wg生成
            with actor_rollout_wg.rwlock.r_lock():
                # 临时切换到当前要生成的agent
                original_agent_idx = actor_rollout_wg.current_agent_idx
                actor_rollout_wg.current_agent_idx = agent_id
                
                output = actor_rollout_wg.generate_sequences(
                    prompts=gen_batch,
                    temperature=self.config.actor_rollout_ref.rollout.temperature,
                    top_p=self.config.actor_rollout_ref.rollout.top_p,
                    max_new_tokens=self.config.data.max_response_length,
                )
                
                # 恢复原始agent索引
                actor_rollout_wg.current_agent_idx = original_agent_idx
            
            # 提取生成的文本（所有agents都用同样的处理逻辑）
            agent_actions = []
            agent_logprobs = []
            for i in range(batch_size):
                if hasattr(output, 'sequences') and output.sequences is not None:
                    generated_ids = output.sequences[i]
                    generated_text = self.tokenizer.decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )
                else:
                    generated_text = f"Response from agent {agent_id}"
                
                # 记录所有agents的生成（保存到日志）
                self.logger.info(f"Step {step}, Agent {agent_id}, Sample {i}: {generated_text[:200]}")
                
                agent_actions.append(generated_text)
                
                # 获取logprobs
                if hasattr(output, 'scores'):
                    logprob = torch.mean(output.scores[i]).item() if output.scores is not None else 0.0
                else:
                    logprob = 0.0
                agent_logprobs.append(logprob)
            
            all_agent_actions.append(agent_actions)
            all_agent_logprobs.append(agent_logprobs)
        
        return all_agent_actions, all_agent_logprobs

    @torch.no_grad()
    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        fixed_agent_wgs: Optional[List] = None,
    ) -> Tuple[List[List[Dict]], np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        执行multi-turn rollout loop - LatentMAS多轮讨论版本
        
        每一轮(round)中:
        - 重新初始化past_kv（区别于原LatentMAS）
        - planner -> critic -> refiner 通过潜在空间通信
        - judger生成文本输出 + 结构化摘要
        - 摘要作为下一轮的前缀传入
        
        Parameters:
            gen_batch: 初始prompt batch
            actor_rollout_wg: 当前训练agent的WorkerGroup（在LatentMAS模式下可能不使用）
            fixed_agent_wgs: 其他固定agent的WorkerGroups
            
        Returns:
            仅返回当前训练agent的trajectory数据
        """
        # 如果未启用LatentMAS，使用原有逻辑
        if not self.use_latent_mas or self.model_wrapper is None:
            return self._vanilla_multi_turn_loop_legacy(
                gen_batch, actor_rollout_wg, fixed_agent_wgs
            )
        
        batch_size = len(gen_batch)
        
        # 初始化轨迹存储
        total_batch_list = [[] for _ in range(batch_size)]
        episode_rewards = np.zeros(batch_size)
        episode_lengths = np.zeros(batch_size, dtype=int)
        
        # 生成轨迹UID
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)])
        
        # 初始化questions
        if hasattr(gen_batch, 'non_tensor_batch') and 'raw_prompt' in gen_batch.non_tensor_batch:
            questions = gen_batch.non_tensor_batch['raw_prompt']
        else:
            questions = [""] * batch_size
        
        # 获取gold answers（如果有）
        gold_answers = [None] * batch_size
        if hasattr(gen_batch, 'non_tensor_batch') and 'gold' in gen_batch.non_tensor_batch:
            gold_answers = gen_batch.non_tensor_batch['gold']
        
        # 用于存储每个sample的结构化摘要（跨轮次传递）
        summary_contexts = [""] * batch_size
        
        # 最终输出文本
        final_texts = [""] * batch_size
        
        self.logger.info(f"[LatentMAS] Starting multi-round loop with {self.max_steps} rounds")
        self.logger.info(f"[LatentMAS] Batch size: {batch_size}, Latent steps: {self.latent_steps}")
        
        # ========== Multi-round loop ==========
        for round_num in range(self.max_steps):
            self.logger.info(f"=== Round {round_num + 1}/{self.max_steps} ===")
            
            # 关键区别：每轮重新初始化past_kv
            past_kv: Optional[Tuple] = None
            round_agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
            
            # ========== 遍历4个agents ==========
            for agent_idx, agent in enumerate(self.agents):
                self.logger.info(f"  Processing Agent: {agent.name} ({agent.role})")
                
                # 构建带有摘要前缀的prompt
                batch_messages = [
                    self._build_agent_prompt_with_summary(
                        role=agent.role,
                        question=questions[i],
                        summary_context=summary_contexts[i],
                        round_num=round_num
                    )
                    for i in range(batch_size)
                ]
                
                # 准备输入
                prompts, input_ids, attention_mask, tokens_batch = \
                    self.model_wrapper.prepare_chat_batch(batch_messages, add_generation_prompt=True)
                
                if agent.role != "judger":
                    # ===== 非judger: 执行latent步骤，更新past_kv =====
                    prev_past_len = _past_length(past_kv)
                    
                    # 可选: 添加think token
                    if self.latent_args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                    else:
                        wrapped_prompts = prompts
                    
                    wrapped_encoded = self.model_wrapper.tokenizer(
                        wrapped_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    wrapped_ids = wrapped_encoded["input_ids"].to(self.model_wrapper.device)
                    wrapped_mask = wrapped_encoded["attention_mask"].to(self.model_wrapper.device)
                    
                    # 执行latent生成，更新past_kv
                    past_kv = self.model_wrapper.generate_latent_batch(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                    )
                    
                    # 记录agent trace
                    for idx in range(batch_size):
                        mask = wrapped_mask[idx].bool()
                        trimmed_ids = wrapped_ids[idx][mask].cpu().tolist()
                        round_agent_traces[idx].append({
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "latent_steps": self.latent_steps,
                            "output": "",  # 非judger没有文本输出
                            "round": round_num,
                        })
                        
                else:
                    # ===== Judger: 生成文本输出 + 结构化摘要 =====
                    past_for_decoding = past_kv if self.latent_steps > 0 else None

                    if self.latent_args.think:
                        judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                    else:
                        judger_prompts = prompts
                    
                    # 调试：检查 prompts 内容
                    self.logger.info(f"[DEBUG Judger] prompts count: {len(prompts)}")
                    for i, p in enumerate(prompts[:2]):  # 只打印前2个
                        self.logger.info(f"[DEBUG Judger] prompts[{i}] length: {len(p)}, preview: {p[:100] if p else 'EMPTY'}...")
                    
                    judger_encoded = self.model_wrapper.tokenizer(
                        judger_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    judger_ids = judger_encoded["input_ids"].to(self.model_wrapper.device)
                    judger_mask = judger_encoded["attention_mask"].to(self.model_wrapper.device)
                    
                    # 生成文本
                    generated_batch, _ = self.model_wrapper.generate_text_batch(
                        judger_ids,
                        judger_mask,
                        max_new_tokens=self.judger_max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        past_key_values=past_for_decoding,
                    )
                    
                    # 处理输出 & 生成结构化摘要
                    for idx in range(batch_size):
                        text_out = generated_batch[idx].strip()
                        final_texts[idx] = text_out
                        
                        # 生成结构化摘要（用于下一轮，最后一轮不需要）
                        if self.enable_structured_summary and round_num < self.max_steps - 1:
                            summary_contexts[idx] = self._parse_structured_summary(text_out)
                        
                        # 记录judger trace
                        round_agent_traces[idx].append({
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                            "structured_summary": summary_contexts[idx] if round_num < self.max_steps - 1 else "",
                            "round": round_num,
                        })
            
            # ========== 记录本轮轨迹数据 ==========
            dones = [round_num >= self.max_steps - 1] * batch_size
            
            for i in range(batch_size):
                # 计算奖励
                if self.reward_fn is not None:
                    reward = self.reward_fn(questions[i], final_texts[i], round_num, dones[i])
                else:
                    # 默认奖励：最后一轮给予1.0
                    reward = 1.0 if dones[i] else 0.0
                
                trajectory_step = {
                    'prompt': questions[i],
                    'action': final_texts[i],
                    'aggregated_action': final_texts[i],
                    'reward': float(reward),  # 确保是 float
                    'done': 1 if dones[i] else 0,  # 转为 int，避免 bool 类型问题
                    'step': int(round_num),  # 确保是 int
                    'traj_uid': traj_uid[i],
                    'active_masks': 1,
                    'all_agent_traces': round_agent_traces[i],
                    'structured_summary': summary_contexts[i],
                    'data_source': 'latent_mas_multi_round',
                }
                
                total_batch_list[i].append(trajectory_step)
                episode_rewards[i] += reward
                episode_lengths[i] = round_num + 1
            
            self.logger.info(f"  Round {round_num + 1} completed. Sample output: {final_texts[0][:100]}...")
            
            # 检查是否全部完成
            if all(dones):
                break
        
        # 构造success字典
        success = {
            'success_rate': np.array([
                1.0 if self._check_answer_correct(questions[i], final_texts[i], gold_answers[i]) else 0.0 
                for i in range(batch_size)
            ])
        }
        
        self.logger.info(f"[LatentMAS] Multi-round loop completed. Success rate: {np.mean(success['success_rate']):.2%}")
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid

    def _vanilla_multi_turn_loop_legacy(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        fixed_agent_wgs: Optional[List] = None,
    ) -> Tuple[List[List[Dict]], np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        原有的multi-turn loop实现（非LatentMAS模式）
        保留用于向后兼容
        """
        batch_size = len(gen_batch)
        
        total_batch_list = [[] for _ in range(batch_size)]
        episode_rewards = np.zeros(batch_size)
        episode_lengths = np.zeros(batch_size, dtype=int)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)])
        
        if hasattr(gen_batch, 'non_tensor_batch') and 'raw_prompt' in gen_batch.non_tensor_batch:
            current_prompts = gen_batch.non_tensor_batch['raw_prompt']
        else:
            current_prompts = [""] * batch_size
        
        for step in range(self.max_steps):
            all_agent_actions, all_agent_logprobs = self.generate_agent_actions(
                gen_batch, 
                actor_rollout_wg,
                fixed_agent_wgs,
                step=step
            )
            
            aggregated_actions = []
            for i in range(batch_size):
                agent_actions_for_env_i = [
                    all_agent_actions[agent_id][i] 
                    for agent_id in range(self.n_agents)
                ]
                agent_logprobs_for_env_i = [
                    all_agent_logprobs[agent_id][i]
                    for agent_id in range(self.n_agents)
                ]
                aggregated_action = self.action_reducer(
                    agent_actions_for_env_i,
                    agent_logprobs_for_env_i
                )
                aggregated_actions.append(aggregated_action)
            
            rewards = []
            dones = [step >= self.max_steps - 1] * batch_size
            
            for i in range(batch_size):
                if self.reward_fn is not None:
                    reward = self.reward_fn(current_prompts[i], aggregated_actions[i], step, dones[i])
                else:
                    reward = 1.0 if dones[i] else 0.0
                rewards.append(reward)
            
            for i in range(batch_size):
                # 单模型架构：使用聚合后的action
                trajectory_step = {
                    'prompt': current_prompts[i] if isinstance(current_prompts, list) else "",
                    'action': aggregated_actions[i],  # 使用聚合后的action
                    'aggregated_action': aggregated_actions[i],
                    'reward': rewards[i],
                    'done': dones[i],
                    'step': step,
                    'traj_uid': traj_uid[i],
                    # 'agent_id' 已移除 - 单模型架构
                    'active_masks': 1,
                    'all_agent_actions': [all_agent_actions[a][i] for a in range(self.n_agents)],
                }
                
                total_batch_list[i].append(trajectory_step)
                episode_rewards[i] += rewards[i]
                episode_lengths[i] = step + 1
            
            next_obs = []
            for i in range(batch_size):
                new_prompt = f"{current_prompts[i]}\nStep {step+1}: {aggregated_actions[i]}"
                next_obs.append(new_prompt)
            current_prompts = next_obs
            
            if all(dones):
                break
        
        success = {
            'success_rate': np.random.random(batch_size) > 0.5
        }
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid

    def gather_rollout_data(
        self,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        traj_uid: np.ndarray,
    ) -> DataProto:
        """
        收集和组织轨迹数据
        """
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)

        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        # Separate tensor and non-tensor data BEFORE collation
        tensor_batch = []  # For numeric data that can be tensorized
        non_tensor_batch = {}  # For strings, lists, etc.
        
        # Initialize non-tensor batch lists
        non_tensor_keys = ['prompt', 'action', 'aggregated_action', 'traj_uid', 
                          'all_agent_traces', 'all_agent_actions', 'structured_summary', 'data_source']
        for key in non_tensor_keys:
            non_tensor_batch[key] = []
        
        for bs in range(batch_size):
            for data in total_batch_list[bs]:
                if data.get('traj_uid') != traj_uid[bs]:
                    print(f"Warning: traj_uid mismatch")
                if data.get('active_masks', 1):
                    # Separate tensor and non-tensor fields
                    tensor_data = {}
                    
                    # Numeric fields for tensor batch
                    tensor_fields = ['reward', 'done', 'step', 'active_masks',
                                   'episode_rewards', 'episode_lengths']
                    
                    for field in tensor_fields:
                        if field in data:
                            tensor_data[field] = data[field]
                        elif field == 'episode_rewards':
                            tensor_data[field] = episode_rewards[bs]
                        elif field == 'episode_lengths':
                            tensor_data[field] = episode_lengths[bs]
                    
                    # Add statistical fields
                    tensor_data['episode_rewards_mean'] = episode_rewards_mean
                    tensor_data['episode_rewards_min'] = episode_rewards_min
                    tensor_data['episode_rewards_max'] = episode_rewards_max
                    tensor_data['episode_lengths_mean'] = episode_lengths_mean
                    tensor_data['episode_lengths_min'] = episode_lengths_min
                    tensor_data['episode_lengths_max'] = episode_lengths_max
                    # 'training_agent_id' 已移除 - 单模型架构
                    
                    # Add success rate
                    for key, value in success_rate.items():
                        tensor_data[key] = value
                    
                    # Add tensor data to batch
                    tensor_batch.append(tensor_data)
                    
                    # Non-tensor fields for non_tensor_batch
                    for key in non_tensor_keys:
                        if key in data:
                            non_tensor_batch[key].append(data[key])
                        else:
                            non_tensor_batch[key].append("")  # Default empty string

        # 调试：检查 tensor_batch 内容
        self.logger.info(f"[gather_rollout_data] tensor_batch length: {len(tensor_batch)}")
        if tensor_batch:
            self.logger.info(f"[gather_rollout_data] tensor_batch[0] keys: {tensor_batch[0].keys()}")
            self.logger.info(f"[gather_rollout_data] tensor_batch[0] sample: {tensor_batch[0]}")    

        # Create DataProto
        # Create DataProto
        if tensor_batch:
            self.logger.info(f"[gather_rollout_data] Creating DataProto with {len(tensor_batch)} items")
            
            # 检查点1: collate_fn 调用
            try:
                collated_tensor_data = collate_fn(tensor_batch)
                self.logger.info(f"[gather_rollout_data] collate_fn succeeded")
                self.logger.info(f"[gather_rollout_data] collated_tensor_data type: {type(collated_tensor_data)}")
                self.logger.info(f"[gather_rollout_data] collated_tensor_data keys: {collated_tensor_data.keys() if isinstance(collated_tensor_data, dict) else 'N/A'}")
                
                # 检查点2: 检查每个字段的类型和形状
                if isinstance(collated_tensor_data, dict):
                    for key, value in collated_tensor_data.items():
                        if hasattr(value, 'shape'):
                            self.logger.info(f"[gather_rollout_data]   {key}: type={type(value).__name__}, shape={value.shape}")
                        else:
                            self.logger.info(f"[gather_rollout_data]   {key}: type={type(value).__name__}, value={value}")
            except Exception as e:
                self.logger.error(f"[gather_rollout_data] collate_fn FAILED: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                collated_tensor_data = {}
            
            # ========== 关键修复：将 numpy.ndarray 转换为 torch.Tensor ==========
            if isinstance(collated_tensor_data, dict):
                keys_to_remove = []  # 记录无法转换的字段
                
                for key, value in collated_tensor_data.items():
                    if isinstance(value, np.ndarray):
                        # 检查 dtype 是否为 object（不支持直接转换）
                        if value.dtype == np.object_:
                            self.logger.warning(f"[gather_rollout_data] {key} has dtype=object, attempting conversion")
                            try:
                                # 尝试转换为 float64
                                value = value.astype(np.float64)
                                self.logger.info(f"[gather_rollout_data] {key} converted from object to float64")
                            except (ValueError, TypeError) as e:
                                self.logger.warning(f"[gather_rollout_data] Cannot convert {key} from object to float64: {e}")
                                # 尝试转换为 int64
                                try:
                                    value = value.astype(np.int64)
                                    self.logger.info(f"[gather_rollout_data] {key} converted from object to int64")
                                except (ValueError, TypeError) as e2:
                                    self.logger.error(f"[gather_rollout_data] Cannot convert {key} to numeric type, skipping: {e2}")
                                    keys_to_remove.append(key)
                                    continue
                        
                        # 转换为 Tensor
                        try:
                            collated_tensor_data[key] = torch.from_numpy(value)
                            self.logger.info(f"[gather_rollout_data] Converted {key} from ndarray({value.dtype}) to Tensor")
                        except Exception as e:
                            self.logger.error(f"[gather_rollout_data] torch.from_numpy failed for {key}: {e}")
                            keys_to_remove.append(key)
                            
                    elif not isinstance(value, torch.Tensor):
                        # 尝试转换其他类型
                        try:
                            collated_tensor_data[key] = torch.tensor(value)
                            self.logger.info(f"[gather_rollout_data] Converted {key} from {type(value).__name__} to Tensor")
                        except Exception as conv_e:
                            self.logger.warning(f"[gather_rollout_data] Cannot convert {key} to Tensor: {conv_e}")
                            keys_to_remove.append(key)
                
                # 移除无法转换的字段
                for key in keys_to_remove:
                    del collated_tensor_data[key]
                    self.logger.warning(f"[gather_rollout_data] Removed unconvertible field: {key}")
            
            # 检查点3: DataProto.from_single_dict 调用
            try:
                if collated_tensor_data:
                    gen_batch_output = DataProto.from_single_dict(collated_tensor_data)
                    self.logger.info(f"[gather_rollout_data] DataProto.from_single_dict succeeded")
                    
                    # 检查点4: 检查 DataProto 的属性
                    self.logger.info(f"[gather_rollout_data] gen_batch_output type: {type(gen_batch_output)}")
                    self.logger.info(f"[gather_rollout_data] gen_batch_output.batch: {gen_batch_output.batch if hasattr(gen_batch_output, 'batch') else 'NO .batch attr'}")
                    if hasattr(gen_batch_output, 'batch') and gen_batch_output.batch is not None:
                        self.logger.info(f"[gather_rollout_data] gen_batch_output.batch type: {type(gen_batch_output.batch)}")
                        if isinstance(gen_batch_output.batch, dict):
                            self.logger.info(f"[gather_rollout_data] gen_batch_output.batch keys: {gen_batch_output.batch.keys()}")
                        elif hasattr(gen_batch_output.batch, 'keys'):
                            self.logger.info(f"[gather_rollout_data] gen_batch_output.batch keys: {gen_batch_output.batch.keys()}")
                else:
                    self.logger.warning(f"[gather_rollout_data] collated_tensor_data is empty, creating empty DataProto")
                    gen_batch_output = DataProto.from_single_dict({})
            except Exception as e:
                self.logger.error(f"[gather_rollout_data] DataProto.from_single_dict FAILED: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                gen_batch_output = DataProto.from_single_dict({})

            # non_tensor_batch 单独挂载
            gen_batch_output.non_tensor_batch = non_tensor_batch
            self.logger.info(f"[gather_rollout_data] non_tensor_batch attached with {len(non_tensor_batch)} keys")
        else:
            self.logger.warning("[gather_rollout_data] tensor_batch is empty!")
            gen_batch_output = DataProto.from_single_dict({})
            gen_batch_output.non_tensor_batch = non_tensor_batch
        
        return gen_batch_output
        
    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        fixed_agent_wgs: Optional[List] = None,
        is_train: bool = True,
    ) -> DataProto:
        """
        主入口函数（单agent训练版本）
        
        Parameters:
            gen_batch: 初始prompt batch
            actor_rollout_wg: 当前训练agent的WorkerGroup
            fixed_agent_wgs: 其他固定agent的WorkerGroups
            is_train: 是否为训练模式
        
        Returns:
            DataProto: 当前训练agent的trajectory数据
        """
        # Training模式：根据配置处理batch
        if is_train and hasattr(self.config, 'env') and hasattr(self.config.env, 'rollout'):
            rollout_n = self.config.env.rollout.get('n', 1)
            if rollout_n > 1:
                gen_batch = gen_batch.repeat(
                    repeat_times=rollout_n,
                    interleave=True
                )
        
        # 执行rollout
        results = self.vanilla_multi_turn_loop(
            gen_batch=gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            fixed_agent_wgs=fixed_agent_wgs,
        )
        
        # 验证数据一致性
        (total_batch_list, total_episode_rewards, total_episode_lengths,
         total_success, total_traj_uid) = results
        
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        
        # 整理并返回
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
        )
        
        print(f"[MultiAgentTrajectoryCollector] Generated {len(total_batch_list)} trajectories (single model, {self.n_agents} roles)")
        
        return gen_batch_output
