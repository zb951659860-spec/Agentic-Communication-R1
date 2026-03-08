#!/usr/bin/env python3
"""
Generic Multi-Agent Rollout Test Script
Reads configuration from YAML file and tests accordingly
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"multi_agent_rollout_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class ConfigBasedMultiAgentTester:
    """Multi-agent rollout tester that reads from config file"""
    
    def __init__(self, config_path: str, override_config: Optional[DictConfig] = None):
        """
        Initialize tester from config file
        
        Args:
            config_path: Path to YAML config file
            override_config: Optional config overrides
        """
        # Load config from file
        self.config = OmegaConf.load(config_path)
        
        # Apply overrides if provided
        if override_config:
            self.config = OmegaConf.merge(self.config, override_config)
        
        # Resolve references in config (e.g., ${data.max_prompt_length})
        OmegaConf.resolve(self.config)
        
        # Extract key configurations
        self.setup_from_config()
        
        # Setup logging
        self.log_dir = self.config.get('log_dir', '../../log/GSM8K_multi_agent_test')
        self.logger = setup_logging(self.log_dir)
        
        # Storage for models and tokenizers
        self.models = []
        self.tokenizers = []
        
    def setup_from_config(self):
        """Extract configurations from loaded config"""
        # Multi-agent configuration
        multi_agent_config = self.config.get('multi_agent', {})
        self.agent_models = multi_agent_config.get('agent_models', [])
        
        # Environment configuration
        env_config = self.config.get('env', {})
        self.n_agents = env_config.get('n_agents', 2)
        self.max_steps = env_config.get('max_steps', 3)
        self.action_reduction = env_config.get('action_reduction', 'majority_vote')
        
        # Data configuration
        data_config = self.config.get('data', {})
        self.test_file = self.expand_path(data_config.get('val_files', ''))
        self.max_response_length = data_config.get('max_response_length', 512)
        
        # Generation parameters (from rollout config)
        rollout_config = self.config.get('actor_rollout_ref', {}).get('rollout', {})
        self.temperature = rollout_config.get('temperature', 0.7)
        self.top_p = rollout_config.get('top_p', 1.0)
        
        # Expand model paths
        self.agent_models = [self.expand_path(path) for path in self.agent_models[:self.n_agents]]
        
        # Validate configuration
        self.validate_config()
        
    def expand_path(self, path: str) -> str:
        """Expand ~ and environment variables in path"""
        if not path:
            return path
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        return path
        
    def validate_config(self):
        """Validate the loaded configuration"""
        if len(self.agent_models) < self.n_agents:
            self.logger.warning(
                f"Config specifies {self.n_agents} agents but only {len(self.agent_models)} model paths provided. "
                f"Adjusting n_agents to {len(self.agent_models)}"
            )
            self.n_agents = len(self.agent_models)
        
        if self.n_agents == 0:
            raise ValueError("No agent models specified in configuration")
        
        # Check if test file exists
        if not os.path.exists(self.test_file):
            raise FileNotFoundError(f"Test data file not found: {self.test_file}")
        
        # Check if model paths exist
        for i, model_path in enumerate(self.agent_models):
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path for agent {i} not found: {model_path}")
    
    def setup_agents(self):
        """Initialize all agent models and tokenizers"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"Using device: {device}")
        
        # Get available GPUs
        if device == "cuda":
            n_gpus = torch.cuda.device_count()
            self.logger.info(f"Available GPUs: {n_gpus}")
        
        for i, model_path in enumerate(self.agent_models):
            self.logger.info(f"Loading Agent {i}: {model_path}")
            
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    padding_side='left'
                )
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Determine device for this model
                target_device = "cuda:0" if device == "cuda" else "cpu"
                
                self.logger.info(f"  Placing Agent {i} on {target_device}")
                
                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=None,  # Don't use auto device map
                    low_cpu_mem_usage=True
                )
                
                # Move model to specific device
                model = model.to(target_device)
                model.eval()
                
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                
                self.logger.info(f"Agent {i} loaded successfully on {target_device}")
                
            except Exception as e:
                self.logger.error(f"Error loading Agent {i}: {str(e)}")
                raise
    
    def load_samples(self, num_samples: int = 1) -> List[Dict[str, Any]]:
        """Load samples from GSM8K dataset"""
        self.logger.info(f"Loading {num_samples} samples from: {self.test_file}")
        
        df = pd.read_parquet(self.test_file)
        self.logger.info(f"Total samples in dataset: {len(df)}")
        
        samples = []
        for i in range(min(num_samples, len(df))):
            row = df.iloc[i]
            
            # Extract question and answer from correct structure
            question = None
            answer = None
            
            # Try to get question from extra_info
            if 'extra_info' in row and isinstance(row['extra_info'], dict):
                question = row['extra_info'].get('question', '')
            
            # Fallback: try prompt field
            if not question and 'prompt' in row:
                prompt = row['prompt']
                if isinstance(prompt, list) and len(prompt) > 0:
                    prompt_content = prompt[0].get('content', '')
                    if "Let's think step by step" in prompt_content:
                        question = prompt_content.split("Let's think step by step")[0].strip()
                    else:
                        question = prompt_content
            
            # Extract answer from reward_model
            if 'reward_model' in row and isinstance(row['reward_model'], dict):
                answer = str(row['reward_model'].get('ground_truth', ''))
            
            sample = {
                'question': question or '',
                'answer': answer or '',
                'index': i
            }
            
            samples.append(sample)
            
        return samples
    
    def format_prompt(self, question: str) -> str:
        """Format the question into a prompt for the model"""
        prompt = f"""Question: {question}

Let's solve this step by step.

Solution:"""
        return prompt
    
    def generate_agent_response(self, agent_idx: int, prompt: str) -> Tuple[str, float]:
        """Generate response from a specific agent"""
        model = self.models[agent_idx]
        tokenizer = self.tokenizers[agent_idx]
        
        # Get the device where this model is located
        model_device = next(model.parameters()).device
        
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_response_length,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode output
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Calculate average logprob
        if hasattr(outputs, 'scores') and outputs.scores:
            logprobs = []
            for i, score in enumerate(outputs.scores):
                if i < len(generated_ids):
                    token_id = generated_ids[i]
                    logprob = torch.log_softmax(score[0], dim=-1)[token_id].item()
                    logprobs.append(logprob)
            avg_logprob = np.mean(logprobs) if logprobs else 0.0
        else:
            avg_logprob = 0.0
        
        return generated_text, avg_logprob
    
    def aggregate_actions(self, actions: List[str], logprobs: List[float]) -> str:
        """Aggregate multiple agent actions using configured strategy"""
        if self.action_reduction == "majority_vote":
            from collections import Counter
            vote_counts = Counter(actions)
            most_common = vote_counts.most_common(1)
            return most_common[0][0] if most_common else actions[0]
        
        elif self.action_reduction == "weighted_vote":
            action_weights = {}
            for action, logprob in zip(actions, logprobs):
                if action not in action_weights:
                    action_weights[action] = 0
                action_weights[action] += np.exp(logprob)
            
            if action_weights:
                best_action = max(action_weights.items(), key=lambda x: x[1])[0]
                return best_action
            return actions[0]
        
        elif self.action_reduction == "first_agent":
            return actions[0] if actions else ""
        
        else:
            return actions[0] if actions else ""
    
    def extract_answer(self, text: str) -> str:
        """Extract numerical answer from generated text"""
        import re
        
        # GSM8K format: #### answer
        gsm8k_pattern = r'####\s*([+-]?\d+(?:\.\d+)?)'
        match = re.search(gsm8k_pattern, text)
        if match:
            return match.group(1)
        
        # Other patterns
        patterns = [
            r'answer is[:\s]+([+-]?\d+(?:\.\d+)?)',
            r'answer[:\s]+([+-]?\d+(?:\.\d+)?)',
            r'=\s*\$?([+-]?\d+(?:\.\d+)?)\s*(?:$|\n)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        
        # Last resort: find the last number
        numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1]
        
        return "N/A"
    
    def perform_multi_agent_rollout(self, sample: Dict[str, Any], sample_idx: int = 0) -> Dict[str, Any]:
        """Perform a multi-agent rollout on a single sample"""
        self.logger.info(f"\nProcessing sample {sample_idx + 1}...")
        
        question = sample.get('question', '')
        gold_answer = sample.get('answer', '')
        
        prompt = self.format_prompt(question)
        
        # Multi-step rollout
        current_prompt = prompt
        all_steps_data = []
        
        for step in range(self.max_steps):
            self.logger.info(f"Step {step + 1}/{self.max_steps}")
            
            # Generate responses from all agents
            step_responses = []
            step_logprobs = []
            
            for agent_idx in range(self.n_agents):
                response, logprob = self.generate_agent_response(agent_idx, current_prompt)
                step_responses.append(response)
                step_logprobs.append(logprob)
                
                self.logger.info(f"  Agent {agent_idx} full response:")
                self.logger.info(f"  {response}")
            
            # Aggregate responses
            aggregated = self.aggregate_actions(step_responses, step_logprobs)
            
            all_steps_data.append({
                'step': step,
                'agent_responses': step_responses,
                'agent_logprobs': step_logprobs,
                'aggregated': aggregated
            })
            
            # Update prompt for next step
            current_prompt = f"{current_prompt}\n{aggregated}\n\nContinue:"
        
        # Extract final answer from last aggregated response
        final_answer = self.extract_answer(all_steps_data[-1]['aggregated'])
        
        # Check correctness
        def normalize_answer(ans):
            return str(ans).strip().replace(',', '').replace('$', '')
        
        is_correct = normalize_answer(final_answer) == normalize_answer(gold_answer)
        
        # Create trajectory
        trajectory = {
            'sample_idx': sample_idx,
            'question': question,
            'gold_answer': gold_answer,
            'final_answer': final_answer,
            'is_correct': is_correct,
            'n_agents': self.n_agents,
            'max_steps': self.max_steps,
            'steps_data': all_steps_data,
            'config_used': {
                'models': self.agent_models,
                'action_reduction': self.action_reduction,
                'temperature': self.temperature,
                'top_p': self.top_p
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return trajectory
    
    def run_test(self, num_samples: int = 1):
        """Run the multi-agent test"""
        self.logger.info("=" * 60)
        self.logger.info("Multi-Agent Test (Config-Based)")
        self.logger.info("=" * 60)
        self.logger.info(f"Configuration loaded from: {self.config_path if hasattr(self, 'config_path') else 'merged config'}")
        self.logger.info(f"Number of agents: {self.n_agents}")
        self.logger.info(f"Max steps: {self.max_steps}")
        self.logger.info(f"Action reduction: {self.action_reduction}")
        self.logger.info("Agent models:")
        for i, model in enumerate(self.agent_models):
            self.logger.info(f"  Agent {i}: {model}")
        self.logger.info("=" * 60)
        
        # Setup agents
        self.setup_agents()
        
        # Load samples
        samples = self.load_samples(num_samples)
        
        # Process samples
        trajectories = []
        correct_count = 0
        
        for idx, sample in enumerate(samples):
            trajectory = self.perform_multi_agent_rollout(sample, idx)
            trajectories.append(trajectory)
            
            if trajectory['is_correct']:
                correct_count += 1
            
            # Save individual trajectory
            traj_file = os.path.join(self.log_dir, f'trajectory_{idx}.json')
            with open(traj_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Sample {idx+1}: {'✓' if trajectory['is_correct'] else '✗'}")
        
        # Calculate accuracy
        accuracy = correct_count / len(trajectories) if trajectories else 0
        
        # Save summary
        summary = {
            'total_samples': len(trajectories),
            'correct': correct_count,
            'accuracy': accuracy,
            'config_file': getattr(self, 'config_path', 'merged'),
            'test_configuration': {
                'n_agents': self.n_agents,
                'max_steps': self.max_steps,
                'action_reduction': self.action_reduction,
                'models': self.agent_models
            },
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(self.log_dir, 'test_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        self.logger.info("=" * 60)
        self.logger.info("Test Complete!")
        self.logger.info(f"Accuracy: {correct_count}/{len(trajectories)} ({accuracy:.2%})")
        self.logger.info(f"Results saved to: {self.log_dir}")
        self.logger.info("=" * 60)
        
        return trajectories, summary


def main():
    parser = argparse.ArgumentParser(description="Config-based Multi-Agent GSM8K Test")
    parser.add_argument("--config", type=str, 
                       default="./verl/trainer/config/ppo_trainer_latent_mas.yaml",
                       help="Path to configuration file")
    parser.add_argument("--num_samples", type=int, default=3, 
                       help="Number of samples to test")
    parser.add_argument("--log_dir", type=str, default=None,
                       help="Override log directory")
    parser.add_argument("--n_agents", type=int, default=None,
                       help="Override number of agents")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Override max steps")
    parser.add_argument("--action_reduction", type=str, default=None,
                       choices=["majority_vote", "weighted_vote", "first_agent"],
                       help="Override action reduction strategy")
    
    args = parser.parse_args()
    
    # Prepare override config
    override_config = {}
    if args.log_dir:
        override_config['log_dir'] = args.log_dir
    if args.n_agents is not None:
        override_config['env'] = {'n_agents': args.n_agents}
    if args.max_steps is not None:
        if 'env' not in override_config:
            override_config['env'] = {}
        override_config['env']['max_steps'] = args.max_steps
    if args.action_reduction:
        if 'env' not in override_config:
            override_config['env'] = {}
        override_config['env']['action_reduction'] = args.action_reduction
    
    override_config = OmegaConf.create(override_config) if override_config else None
    
    # Create tester
    tester = ConfigBasedMultiAgentTester(args.config, override_config)
    tester.config_path = args.config  # Store for logging
    
    # Run test
    trajectories, summary = tester.run_test(num_samples=args.num_samples)
    
    print(f"\nTest completed! Results saved to: {tester.log_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
