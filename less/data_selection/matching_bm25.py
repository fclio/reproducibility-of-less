import argparse
import os
import json
import torch
import random
from tqdm import tqdm
import pandas as pd

argparser = argparse.ArgumentParser(
    description='Script for selecting data using random scores.'
)
argparser.add_argument('--train_data_path', type=str, required=True,
                       help='The path to the training data directory or file.')
argparser.add_argument('--train_file_names', type=str, nargs='+', required=True,
                       help='The names of the training data files.')
argparser.add_argument('--eval_data_path', type=str, required=True,
                       help='The path to the evaluation data directory or file (not used here).')
argparser.add_argument('--target_task_names', type=str, nargs='+', required=True,
                       help='The names of the target tasks (not used for scoring, but for directory structure).')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output directory.')

args = argparser.parse_args()

# Number of subtasks for each target task (not directly used for random, but kept for consistency)
N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_jsonl(file_path):
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Loading {file_path}"):
                data = json.loads(line)
                # Extract content as needed
                content_pieces = []
                messages = data.get('messages', [])
                for message in messages:
                    message_content = message.get('content', '').strip()
                    if message_content:
                        content_pieces.append(message_content)
                content = " ".join(content_pieces)
                if content.strip():
                    data_list.append(content)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return data_list

def load_json(file_path):
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
            # Adjust the following based on your data structure
            data_entries = data_dict.get('data', [])
            for entry in tqdm(data_entries, desc=f"Loading {file_path}"):
                # For simplicity, serialize the entry to a string
                content = json.dumps(entry)
                if content.strip():
                    data_list.append(content)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return data_list

def load_training_data(train_file_name):
    """Load lines from a training file or directory corresponding to train_file_name."""
    training_path = os.path.join(args.train_data_path, train_file_name)
    training_data = []

    if os.path.isdir(training_path):
        # Load all files in the directory (jsonl/json)
        for root, dirs, files in os.walk(training_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.jsonl'):
                    training_data.extend(load_jsonl(file_path))
                elif file.endswith('.json'):
                    training_data.extend(load_json(file_path))
    else:
        # Single file case
        if training_path.endswith('.jsonl'):
            training_data = load_jsonl(training_path)
        elif training_path.endswith('.json'):
            training_data = load_json(training_path)
        else:
            # If not json/jsonl, just read lines
            with open(training_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        training_data.append(line)

    return training_data

# Set a seed if reproducibility is desired
# random.seed(42)
# torch.manual_seed(42)

for target_task_name in args.target_task_names:
    for train_file_name in args.train_file_names:
        print(f"Processing random scores for training file {train_file_name} and target task {target_task_name}")
        # Load the training data lines
        training_data = load_training_data(train_file_name)
        num_samples = len(training_data)
        print(f"Loaded {num_samples} training documents for {train_file_name}.")

        # Generate random scores for each training sample
        influence_score = torch.rand(num_samples, dtype=torch.float32)

        # Save influence scores
        output_dir = os.path.join(args.output_path, target_task_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"{train_file_name}_random_influence_score.pt")
        torch.save(influence_score, output_file)
        print(f"Saved random influence score to {output_file}")
