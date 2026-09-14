#!/usr/bin/env python3
"""
Test script to verify GSM8K data loading is working correctly
"""

import pandas as pd
import json
from typing import Dict, Any, List

def test_data_loading(test_file: str):
    """Test if data is being loaded correctly"""
    
    print("=" * 60)
    print("Testing GSM8K Data Loading")
    print("=" * 60)
    
    # Load the dataset
    df = pd.read_parquet(test_file)
    print(f"Total samples in dataset: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Test loading first 3 samples
    for i in range(min(3, len(df))):
        print(f"\n--- Testing Sample {i+1} ---")
        row = df.iloc[i]
        
        # Extract question
        question = None
        
        # Method 1: from extra_info
        if 'extra_info' in row and isinstance(row['extra_info'], dict):
            question = row['extra_info'].get('question', '')
            print(f"✓ Found question in extra_info")
        
        # Method 2: from prompt (fallback)
        if not question and 'prompt' in row:
            prompt = row['prompt']
            if isinstance(prompt, list) and len(prompt) > 0:
                prompt_content = prompt[0].get('content', '')
                if "Let's think step by step" in prompt_content:
                    question = prompt_content.split("Let's think step by step")[0].strip()
                else:
                    question = prompt_content
                print(f"✓ Found question in prompt field")
        
        # Extract answer
        answer = ''
        if 'reward_model' in row and isinstance(row['reward_model'], dict):
            answer = str(row['reward_model'].get('ground_truth', ''))
            print(f"✓ Found answer in reward_model: {answer}")
        
        # Extract solution
        solution = ''
        if 'extra_info' in row and isinstance(row['extra_info'], dict):
            solution = row['extra_info'].get('answer', '')
            print(f"✓ Found solution in extra_info")
        
        # Display extracted data
        print(f"\nExtracted Data:")
        print(f"  Question: {question[:100] if question else '[EMPTY]'}...")
        print(f"  Answer: {answer}")
        print(f"  Has solution: {'Yes' if solution else 'No'}")
        
        # Test answer extraction from solution
        if solution:
            import re
            gsm8k_pattern = r'####\s*([+-]?\d+(?:\.\d+)?)'
            match = re.search(gsm8k_pattern, solution)
            if match:
                extracted = match.group(1)
                print(f"  Extracted answer from solution: {extracted}")
                print(f"  Matches ground truth: {extracted == answer}")
    
    print("\n" + "=" * 60)
    print("Data Loading Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_file = "../../data/GSM8K/test.parquet"
    test_data_loading(test_file)
