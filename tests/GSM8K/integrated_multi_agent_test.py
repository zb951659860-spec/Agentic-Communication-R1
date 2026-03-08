#!/usr/bin/env python3
"""
Integrated Multi-Agent Test Script
Uses MultiAgentTrajectoryCollector from rollout_loop.py
with configuration and logging from config_based_multi_agent_test.py

支持两种模式：
1. Legacy模式：使用MockActorRolloutWG进行测试
2. LatentMAS模式：使用内置ModelWrapper进行潜在空间通信
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig

# Import the MultiAgentTrajectoryCollector
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from agent_system.multi_turn_rollout.rollout_loop import MultiAgentTrajectoryCollector


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"multi_agent_rollout_{timestamp}.log")
    
    # Create a specific logger instead of using root logger
    logger = logging.getLogger('integrated_multi_agent_test')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Also print where the log file is
    print(f"Log file created: {log_file}")
    
    return logger


class IntegratedMultiAgentTester:
    """
    Integrated tester using MultiAgentTrajectoryCollector with config-based setup
    
    支持两种模式:
    - use_latent_mas=False: 使用MockActorRolloutWG (legacy模式)
    - use_latent_mas=True: 使用内置ModelWrapper进行LatentMAS测试
    """
    
    def __init__(self, config_path: str, override_config: Optional[DictConfig] = None):
        """
        Initialize tester from config file
        
        Args:
            config_path: Path to YAML config file
            override_config: Optional config overrides
        """
        # Load config from file
        self.config = OmegaConf.load(config_path)
        self.config_path = config_path
        
        # Apply overrides if provided
        if override_config:
            self.config = OmegaConf.merge(self.config, override_config)
        
        # Resolve references in config
        OmegaConf.resolve(self.config)
        
        # Setup logging
        self.log_dir = self.config.get('log_dir', '../../log/GSM8K_integrated_test')
        self.logger = setup_logging(self.log_dir)
        
        # Extract configurations
        self.setup_from_config()
        
        # Initialize components
        self.tokenizer = None
        self.processor = None
        self.trajectory_collector = None
        self.actor_rollout_wg = None
        
    def setup_from_config(self):
        """Extract configurations from loaded config"""
        # Multi-agent configuration
        multi_agent_config = self.config.get('multi_agent', {})
        self.agent_models = multi_agent_config.get('agent_models', [])
        
        # Environment configuration
        env_config = self.config.get('env', {})
        self.n_agents = 4  # 固定为4个agent（LatentMAS风格）
        self.max_steps = env_config.get('max_steps', 3) if env_config else self.config.get('max_steps', 3)
        
        # LatentMAS configuration
        self.use_latent_mas = self.config.get('use_latent_mas', False)
        self.model_name = self.config.get('model_name', None)
        self.latent_steps = self.config.get('latent_steps', 3)
        self.task = self.config.get('task', 'gsm8k')
        
        # Data configuration
        data_config = self.config.get('data', {})
        self.test_file = self.expand_path(data_config.get('val_files', '')) if data_config else ''
        
        # Expand model paths (only needed for legacy mode)
        if not self.use_latent_mas:
            self.agent_models = [self.expand_path(path) for path in self.agent_models[:self.n_agents]]
        
        # Validate
        self.validate_config()
        
        self.logger.info(f"Configuration loaded:")
        self.logger.info(f"  use_latent_mas: {self.use_latent_mas}")
        self.logger.info(f"  model_name: {self.model_name}")
        self.logger.info(f"  n_agents: {self.n_agents}")
        self.logger.info(f"  max_steps: {self.max_steps}")
        self.logger.info(f"  latent_steps: {self.latent_steps}")
        self.logger.info(f"  task: {self.task}")
        
    def expand_path(self, path: str) -> str:
        """Expand ~ and environment variables in path"""
        if not path:
            return path
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        return path
    
    def validate_config(self):
        """Validate the loaded configuration"""
        if self.use_latent_mas:
            # LatentMAS模式需要model_name
            if not self.model_name:
                raise ValueError("use_latent_mas=True requires 'model_name' in config")
            self.logger.info(f"[LatentMAS Mode] Using model: {self.model_name}")
        else:
            # Legacy模式需要agent_models
            if len(self.agent_models) < self.n_agents:
                self.logger.warning(f"Adjusting n_agents from {self.n_agents} to {len(self.agent_models)}")
                self.n_agents = len(self.agent_models)
            
            for i, model_path in enumerate(self.agent_models):
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Model path for agent {i} not found: {model_path}")
        
        # 检查测试文件（如果指定）
        if self.test_file and not os.path.exists(self.test_file):
            self.logger.warning(f"Test data file not found: {self.test_file}, will use synthetic data")
            self.test_file = None
    
    def setup_tokenizer_and_processor(self):
        """Setup tokenizer and processor"""
        from transformers import AutoTokenizer
        
        self.logger.info("Setting up tokenizer and processor...")
        
        # 确定使用哪个模型路径
        if self.use_latent_mas:
            base_model_path = self.model_name
        else:
            base_model_path = self.agent_models[0] if self.agent_models else None
        
        if not base_model_path:
            raise ValueError("No model path available for tokenizer initialization")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                padding_side='left'
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info(f"Tokenizer loaded from: {base_model_path}")
            
            # Processor is optional (for multimodal models)
            self.processor = None
            
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {str(e)}")
            raise
    
    def setup_trajectory_collector(self):
        """Initialize MultiAgentTrajectoryCollector with config"""
        self.logger.info("Setting up MultiAgentTrajectoryCollector...")
        
        # Create reward function (simplified for testing)
        def simple_reward_fn(prompt, action, step, done):
            if done:
                # Simple reward: check if answer contains a number in boxed format
                import re
                boxed = re.findall(r'\\boxed\{([^}]*)\}', action)
                if boxed:
                    return 1.0
                numbers = re.findall(r'\d+', action)
                return 0.5 if numbers else 0.0
            return 0.0
        
        # Initialize the trajectory collector (单模型架构)
        self.trajectory_collector = MultiAgentTrajectoryCollector(
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            reward_fn=simple_reward_fn,
        )
        
        self.logger.info(f"MultiAgentTrajectoryCollector initialized (single model architecture)")
        self.logger.info(f"  use_latent_mas: {self.trajectory_collector.use_latent_mas}")
        self.logger.info(f"  n_roles: {self.trajectory_collector.n_agents}")
    
    def create_mock_actor_rollout_wg(self):
        """
        Create a mock ActorRolloutWorkerGroup for testing (Legacy mode only)
        This simulates the actor_rollout_wg used in training
        """
        if self.use_latent_mas:
            self.logger.info("[LatentMAS Mode] Skipping MockActorRolloutWG creation - using internal ModelWrapper")
            self.actor_rollout_wg = None
            return
        
        from types import SimpleNamespace
        import threading
        
        class MockRWLock:
            def __init__(self):
                self.lock = threading.RLock()
            
            def r_lock(self):
                return self.lock
        
        class MockActorRolloutWG:
            def __init__(self, models, tokenizers, config):
                self.models = models
                self.tokenizers = tokenizers
                self.config = config
                self.rwlock = MockRWLock()
                self.current_agent_idx = config.get('current_agent_idx', 0)
            
            def generate_sequences(self, prompts, temperature=0.7, top_p=0.9, max_new_tokens=512):
                """Generate sequences using the current training agent's model"""
                model = self.models[self.current_agent_idx]
                tokenizer = self.tokenizers[self.current_agent_idx]
                
                # Get device
                device = next(model.parameters()).device
                
                # Handle DataProto input
                batch_size = len(prompts)
                
                # Extract data from DataProto
                if hasattr(prompts, 'batch'):
                    input_ids = prompts.batch.get('input_ids')
                    attention_mask = prompts.batch.get('attention_mask')
                    if input_ids is not None:
                        inputs = {
                            'input_ids': input_ids.to(device),
                            'attention_mask': attention_mask.to(device) if attention_mask is not None else None
                        }
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                do_sample=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id
                            )
                        result = SimpleNamespace()
                        result.sequences = outputs
                        result.scores = None
                        return result
                
                # Fallback to text-based generation
                if hasattr(prompts, 'non_tensor_batch'):
                    prompt_texts = prompts.non_tensor_batch.get('raw_prompt', [])
                    if not prompt_texts:
                        prompt_texts = [""] * batch_size
                else:
                    prompt_texts = prompts if isinstance(prompts, list) else [prompts]
                
                all_outputs = []
                for prompt_text in prompt_texts:
                    if isinstance(prompt_text, list):
                        prompt_text = prompt_text[0] if prompt_text else ""
                    
                    inputs = tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024,
                        padding=True
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                    
                    generated_only = outputs[0][inputs['input_ids'].shape[1]:]
                    all_outputs.append(generated_only)
                
                result = SimpleNamespace()
                result.sequences = all_outputs
                result.scores = None
                return result
        
        # Load models for the mock WG
        self.logger.info("Loading models for mock ActorRolloutWG...")
        
        from transformers import AutoModelForCausalLM
        
        models = []
        tokenizers = []
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for i, model_path in enumerate(self.agent_models):
            self.logger.info(f"Loading model {i}: {model_path}")
            
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side='left'
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizers.append(tokenizer)
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            )
            model = model.to(f"cuda:0" if device == "cuda" else "cpu")
            model.eval()
            models.append(model)
        
        self.actor_rollout_wg = MockActorRolloutWG(models, tokenizers, self.config)
        self.logger.info("Mock ActorRolloutWG created")
    
    def load_test_data(self, num_samples: int) -> Any:
        """Load test data and create DataProto batch"""
        from verl import DataProto
        
        self.logger.info(f"Loading {num_samples} samples...")
        
        samples = []
        gold_answers = []
        
        # 尝试从文件加载
        if self.test_file and os.path.exists(self.test_file):
            self.logger.info(f"Loading from file: {self.test_file}")
            df = pd.read_parquet(self.test_file)
            
            for i in range(min(num_samples, len(df))):
                row = df.iloc[i]
                
                # Extract question
                question = None
                gold = None
                
                if 'extra_info' in row and isinstance(row['extra_info'], dict):
                    question = row['extra_info'].get('question', '')
                    gold = row['extra_info'].get('answer', None)
                
                if not question and 'prompt' in row:
                    prompt = row['prompt']
                    if isinstance(prompt, list) and len(prompt) > 0:
                        question = prompt[0].get('content', '')
                
                # Try to extract gold answer from different formats
                if gold is None and 'answer' in row:
                    gold = str(row['answer'])
                
                samples.append(question or f"Test question {i+1}")
                gold_answers.append(gold)
        else:
            # Use synthetic GSM8K-style questions for testing
            self.logger.info("Using synthetic test data")
            synthetic_questions = [
                "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
                "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            ]
            synthetic_answers = ["18", "3", "70000"]
            
            for i in range(num_samples):
                idx = i % len(synthetic_questions)
                samples.append(synthetic_questions[idx])
                gold_answers.append(synthetic_answers[idx])
        
        # Create DataProto batch
        tokenized_samples = []
        for sample in samples:
            inputs = self.tokenizer(
                sample,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=512
            )
            tokenized_samples.append(inputs)
        
        batch_dict = {
            'input_ids': torch.cat([s['input_ids'] for s in tokenized_samples], dim=0),
            'attention_mask': torch.cat([s['attention_mask'] for s in tokenized_samples], dim=0),
            'position_ids': torch.arange(512).unsqueeze(0).repeat(len(samples), 1),
        }
        
        full_batch = DataProto.from_single_dict(batch_dict)
        
        full_batch.non_tensor_batch = {        
            'raw_prompt': samples,
            'gold': gold_answers,  # 添加gold答案用于评估
            'data_source': [self.task] * len(samples)
        }
        
        gen_batch = full_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt", "gold", "data_source"]
        )
        
        self.logger.info(f"Loaded {len(samples)} samples")
        return gen_batch
    
    def run_test(self, num_samples: int = 1):
        """Run the integrated test using MultiAgentTrajectoryCollector"""
        self.logger.info("=" * 60)
        self.logger.info("Integrated Multi-Agent Test")
        self.logger.info(f"Mode: {'LatentMAS' if self.use_latent_mas else 'Legacy'}")
        self.logger.info("=" * 60)
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Number of agents: {self.n_agents}")
        self.logger.info(f"Max steps (rounds): {self.max_steps}")
        if self.use_latent_mas:
            self.logger.info(f"Model: {self.model_name}")
            self.logger.info(f"Latent steps: {self.latent_steps}")
        else:
            self.logger.info("Agent models:")
            for i, model in enumerate(self.agent_models):
                self.logger.info(f"  Agent {i}: {model}")
        self.logger.info("=" * 60)
        
        # Setup components
        self.setup_tokenizer_and_processor()
        self.setup_trajectory_collector()
        self.create_mock_actor_rollout_wg()
        
        # Load test data
        gen_batch = self.load_test_data(num_samples)
        
        # Run rollout using MultiAgentTrajectoryCollector
        self.logger.info("\nStarting rollout with MultiAgentTrajectoryCollector...")
        
        try:
            # Call the multi_turn_loop method
            gen_batch_output = self.trajectory_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                fixed_agent_wgs=None,
                is_train=False  # Testing mode
            )
            
            self.logger.info("Rollout completed successfully!")

            # Debug output
            print("non_tensor keys:", getattr(gen_batch_output, "non_tensor_batch", {}).keys())
            nt = getattr(gen_batch_output, "non_tensor_batch", {})
            print("has action:", "action" in nt, "has aggregated_action:", "aggregated_action" in nt)
            if "aggregated_action" in nt and nt["aggregated_action"]:
                print("sample aggregated_action[0][:200]:", str(nt["aggregated_action"][0])[:200])

            # Process and save results
            results = self.process_results(gen_batch_output)
            
            # Save summary
            summary = {
                'num_samples': num_samples,
                'n_agents': self.n_agents,
                'max_steps': self.max_steps,
                'use_latent_mas': self.use_latent_mas,
                'model_name': self.model_name if self.use_latent_mas else None,
                'latent_steps': self.latent_steps if self.use_latent_mas else None,
                'config_file': self.config_path,
                'agent_models': self.agent_models if not self.use_latent_mas else [],
                'timestamp': datetime.now().isoformat(),
                'results': results
            }
            
            summary_file = os.path.join(self.log_dir, 'integrated_test_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"\nResults saved to: {self.log_dir}")
            self.logger.info("Test completed successfully!")
            
            return gen_batch_output, summary
            
        except Exception as e:
            self.logger.error(f"Error during rollout: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def process_results(self, gen_batch_output):
        """Process the output from MultiAgentTrajectoryCollector"""
        results = []
        
        self.logger.info("Processing results from MultiAgentTrajectoryCollector...")
        
        # Process tensor batch data

        # 检查点: 详细检查 gen_batch_output 的结构
        self.logger.info(f"[run_test] gen_batch_output type: {type(gen_batch_output)}")
        self.logger.info(f"[run_test] gen_batch_output dir: {[attr for attr in dir(gen_batch_output) if not attr.startswith('_')]}")
        
        if hasattr(gen_batch_output, 'batch'):
            self.logger.info(f"[run_test] gen_batch_output.batch: {gen_batch_output.batch}")
            self.logger.info(f"[run_test] gen_batch_output.batch is None: {gen_batch_output.batch is None}")
        else:
            self.logger.info(f"[run_test] gen_batch_output has NO 'batch' attribute")
        
        if hasattr(gen_batch_output, 'non_tensor_batch'):
            self.logger.info(f"[run_test] gen_batch_output.non_tensor_batch keys: {gen_batch_output.non_tensor_batch.keys() if gen_batch_output.non_tensor_batch else 'None'}")
            
        if hasattr(gen_batch_output, 'batch') and gen_batch_output.batch is not None:
            batch_data = gen_batch_output.batch

            if isinstance(batch_data, dict):
                self.logger.info(f"Batch data keys: {batch_data.keys()}")
                
                if 'episode_rewards' in batch_data:
                    rewards = batch_data['episode_rewards']
                    self.logger.info(f"Episode rewards: {rewards}")
                    results.append({'episode_rewards': rewards})
                
                if 'episode_lengths' in batch_data:
                    lengths = batch_data['episode_lengths']
                    self.logger.info(f"Episode lengths: {lengths}")
                    results.append({'episode_lengths': lengths})
                
                for key in ['episode_rewards_mean', 'episode_rewards_min', 'episode_rewards_max', 'success_rate']:
                    if key in batch_data:
                        self.logger.info(f"{key}: {batch_data[key]}")
            else:
                self.logger.info(f"Batch data type: {type(batch_data)}")
                if hasattr(batch_data, 'episode_rewards'):
                    rewards = batch_data.episode_rewards
                    self.logger.info(f"Episode rewards (from attr): {rewards}")
                    results.append({'episode_rewards': rewards})
        else:
            self.logger.info("No tensor batch data found")
        
        # Process non-tensor batch data
        if hasattr(gen_batch_output, 'non_tensor_batch') and gen_batch_output.non_tensor_batch is not None:
            non_tensor = gen_batch_output.non_tensor_batch
            
            if isinstance(non_tensor, dict):
                self.logger.info(f"Non-tensor batch keys: {non_tensor.keys()}")
                
                if 'action' in non_tensor and non_tensor['action']:
                    self.logger.info(f"Sample action: {non_tensor['action'][0][:100] if non_tensor['action'][0] else 'None'}...")
                if 'prompt' in non_tensor and non_tensor['prompt']:
                    self.logger.info(f"Sample prompt: {non_tensor['prompt'][0][:100] if non_tensor['prompt'][0] else 'None'}...")
                
                results.append({'non_tensor_data': {
                    'num_samples': len(non_tensor.get('action', [])),
                    'has_prompts': 'prompt' in non_tensor,
                    'has_actions': 'action' in non_tensor,
                    'has_traj_uid': 'traj_uid' in non_tensor,
                    'has_all_agent_traces': 'all_agent_traces' in non_tensor,
                    'has_structured_summary': 'structured_summary' in non_tensor,
                }})
            else:
                self.logger.info(f"Non-tensor data type: {type(non_tensor)}")
        else:
            self.logger.info("No non-tensor batch data found")

        if results:
            self.logger.info("=" * 40)
            self.logger.info("SUMMARY OF RESULTS:")
            for r in results:
                self.logger.info(f"  {r}")
        
        # 显示详细的agent输出（适配LatentMAS格式）
        if hasattr(gen_batch_output, 'non_tensor_batch') and gen_batch_output.non_tensor_batch:
            non_tensor = gen_batch_output.non_tensor_batch
            
            # 处理 all_agent_traces（LatentMAS格式）
            if 'all_agent_traces' in non_tensor and non_tensor['all_agent_traces']:
                self.logger.info("=" * 40)
                self.logger.info("DETAILED AGENT TRACES (LatentMAS Mode):")
                
                # 获取第一个样本的traces
                for step_idx, traces in enumerate(non_tensor['all_agent_traces'][:self.max_steps]):
                    if traces:  # traces是一个list of dict
                        self.logger.info(f"\n--- Round {step_idx + 1} ---")
                        for trace in traces:
                            if isinstance(trace, dict):
                                agent_name = trace.get('name', 'Unknown')
                                agent_role = trace.get('role', 'Unknown')
                                output = trace.get('output', '')
                                latent_steps = trace.get('latent_steps', 'N/A')
                                
                                self.logger.info(f"\nAgent: {agent_name} ({agent_role})")
                                if output:
                                    self.logger.info(f"Output: {output[:300]}...")
                                else:
                                    self.logger.info(f"Latent steps: {latent_steps} (no text output)")
                                
                                if 'structured_summary' in trace and trace['structured_summary']:
                                    self.logger.info(f"Summary: {trace['structured_summary'][:200]}...")
            
            # 处理 all_agent_actions（Legacy格式）
            elif 'all_agent_actions' in non_tensor and non_tensor['all_agent_actions']:
                self.logger.info("=" * 40)
                self.logger.info("DETAILED AGENT OUTPUTS (Legacy Mode):")
                
                num_samples_per_step = len(non_tensor['all_agent_actions']) // self.max_steps if self.max_steps > 0 else len(non_tensor['all_agent_actions'])
                
                for step in range(min(self.max_steps, 2)):  # 只显示前2轮
                    self.logger.info(f"\n--- Step {step} ---")
                    
                    sample_idx = step * num_samples_per_step if num_samples_per_step > 0 else step
                    if sample_idx < len(non_tensor['all_agent_actions']):
                        agent_actions = non_tensor['all_agent_actions'][sample_idx]
                        
                        if step == 0 and 'prompt' in non_tensor and sample_idx < len(non_tensor['prompt']):
                            prompt = non_tensor['prompt'][sample_idx]
                            if prompt:
                                self.logger.info(f"Question: {prompt[:200]}...")
                        
                        for agent_idx, action in enumerate(agent_actions):
                            self.logger.info(f"\nAgent {agent_idx} response:")
                            self.logger.info(f"{action[:300] if action else 'Empty'}...")
                            
                            import re
                            numbers = re.findall(r'\\boxed\{([^}]*)\}', action)
                            if not numbers:
                                numbers = re.findall(r'\d+', action)
                            self.logger.info(f"Extracted answer: {numbers[-1] if numbers else 'No number found'}")
            
            # 显示统计信息
            self.logger.info("=" * 40)
            self.logger.info("STATISTICS:")
            self.logger.info(f"Total trajectory items: {len(non_tensor.get('action', []))}")
            self.logger.info(f"Number of rounds: {self.max_steps}")
            self.logger.info(f"Number of agents: {self.n_agents}")
            self.logger.info(f"Mode: {'LatentMAS' if self.use_latent_mas else 'Legacy'}")
                
        return results


def main():
    parser = argparse.ArgumentParser(description="Integrated Multi-Agent Test")
    parser.add_argument("--config", type=str,
                       default="./verl/trainer/config/ppo_trainer_latent_mas.yaml",
                       help="Path to configuration file")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of samples to test")
    parser.add_argument("--log_dir", type=str, default=None,
                       help="Override log directory")
    # --current_agent_idx 已移除 - 单模型架构不需要
    
    # LatentMAS specific arguments
    parser.add_argument("--use_latent_mas", action="store_true",
                       help="Enable LatentMAS mode")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name for LatentMAS mode (e.g., 'Qwen/Qwen3-4B')")
    parser.add_argument("--latent_steps", type=int, default=3,
                       help="Number of latent steps for LatentMAS")
    parser.add_argument("--max_steps", type=int, default=3,
                       help="Number of discussion rounds")
    parser.add_argument("--task", type=str, default="gsm8k",
                       help="Task type (gsm8k, aime2024, etc.)")
    
    args = parser.parse_args()
    
    # Prepare override config
    override_config = {}
    if args.log_dir:
        override_config['log_dir'] = args.log_dir
    if args.use_latent_mas:
        override_config['use_latent_mas'] = True
    if args.model_name:
        override_config['model_name'] = args.model_name
    if args.latent_steps:
        override_config['latent_steps'] = args.latent_steps
    if args.max_steps:
        override_config['max_steps'] = args.max_steps
    if args.task:
        override_config['task'] = args.task
    
    override_config = OmegaConf.create(override_config) if override_config else None
    
    # Create and run tester
    tester = IntegratedMultiAgentTester(args.config, override_config)
    
    try:
        gen_batch_output, summary = tester.run_test(num_samples=args.num_samples)
        print("\n" + "="*60)
        print("Test Summary:")
        print(json.dumps(summary, indent=2, default=str))
        print("="*60)
    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
