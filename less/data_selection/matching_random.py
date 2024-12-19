import argparse
import os
import json
import torch
import random
from tqdm import tqdm
import pandas as pd

argparser = argparse.ArgumentParser(
    description='Script for selecting data using random scores, following the exact logic of matching_bm25.py'
)
argparser.add_argument('--train_data_path', type=str, required=True,
                       help='The path to the training data directory or file.')
argparser.add_argument('--train_file_names', type=str, nargs='+', required=True,
                       help='The names of the training data files.')
argparser.add_argument('--eval_data_path', type=str, required=True,
                       help='The path to the evaluation data directory or file.')
argparser.add_argument('--target_task_names', type=str, nargs='+', required=True,
                       help='The names of the target tasks.')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output directory.')

args = argparser.parse_args()

# Number of subtasks for each target task
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
            data_entries = data_dict.get('data', [])
            for entry in tqdm(data_entries, desc=f"Loading {file_path}"):
                content = json.dumps(entry)
                if content.strip():
                    data_list.append(content)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return data_list

def load_tydiqa_data(tydiqa_dir):
    dev_dir = os.path.join(tydiqa_dir, 'dev')
    if 'tydiqa-one-shot-zh.json' in os.listdir(dev_dir):
        file_name = 'tydiqa-one-shot-zh.json'
    elif 'tydiqa-one-shot.json' in os.listdir(dev_dir):
        file_name = 'tydiqa-one-shot.json'
    else:
        print(f"Error: Neither 'tydiqa-one-shot.json' nor 'tydiqa-one-shot-zh.json' found in {dev_dir}")
        return []

    file_path = os.path.join(dev_dir, file_name)

    with open(file_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    data_list = []
    for lang in examples:
        for example in examples[lang]:
            context = example.get("context", "")
            question = example.get("question", "")
            content = context + " " + question
            data_list.append(content)

    return data_list

def load_mmlu_data(mmlu_dir):
    data_list = []
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(mmlu_dir, "test"))
            if "_test.csv" in f
        ]
    )

    for subject in subjects:
        dev_file = os.path.join(mmlu_dir, "dev", subject + "_dev.csv")
        if not os.path.exists(dev_file):
            continue

        dev_df = pd.read_csv(dev_file, header=None)
        for idx in range(len(dev_df)):
            question = dev_df.iloc[idx, 0]
            choices = dev_df.iloc[idx, 1:5].tolist()
            content = question + ' ' + ' '.join(choices)
            data_list.append(content)

    return data_list

def load_bbh_data(bbh_dir):
    data_list = []
    file_path = os.path.join(bbh_dir, 'bbh-three-shot.json')
    if not os.path.exists(file_path):
        print(f"BBH data file not found at {file_path}")
        return data_list

    with open(file_path, 'r', encoding='utf-8') as f:
        bbh_few_shot_examples = json.load(f)

    for task in bbh_few_shot_examples:
        few_shot_exs = bbh_few_shot_examples[task]
        stuff = few_shot_exs.split("\n\n")
        exes = stuff[-3:]
        task_prompt = "\n\n".join(stuff[:-3])

        for i in range(len(exes)):
            target_ex = exes[i]
            other_exes = exes[:i] + exes[i+1:]
            icl = "\n\n".join(other_exes)
            question, answer = target_ex.split("\nA:")
            content = task_prompt.strip() + "\n\n" + icl + "\n\n" + question
            data_list.append(content)
    return data_list

def load_training_data(train_file_name):
    training_path = os.path.join(args.train_data_path, train_file_name)
    training_data = []

    if os.path.isdir(training_path):
        for root, dirs, files in os.walk(training_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.jsonl'):
                    training_data.extend(load_jsonl(file_path))
                elif file.endswith('.json'):
                    training_data.extend(load_json(file_path))
    else:
        if training_path.endswith('.jsonl'):
            training_data = load_jsonl(training_path)
        elif training_path.endswith('.json'):
            training_data = load_json(training_path)
        else:
            with open(training_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        training_data.append(line)

    training_data = [doc for doc in training_data if doc.strip()]
    return training_data

##################################
# NEW FUNCTIONS FOR fiqa, nfcorpus, scifact, vihealthqa
##################################

def load_fiqa_data(fiqa_dir):
    """
    Load the FIQA dataset from fiqa_dev.jsonl and extract content.
    Format is similar to scifact.
    """
    file_path = os.path.join(fiqa_dir, 'fiqa_dev.jsonl')
    if not os.path.exists(file_path):
        print(f"FIQA data file not found at {file_path}")
        return []

    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {file_path}"):
            data = json.loads(line)
            messages = data.get('messages', [])
            content_pieces = []
            for message in messages:
                message_content = message.get('content', '').strip()
                if message_content:
                    content_pieces.append(message_content)
            content = " ".join(content_pieces)
            if content.strip():
                data_list.append(content)
    return data_list

def load_nfcorpus_data(nfcorpus_dir):
    """
    Load the NFCorpus dataset from nfcorpus_dev.jsonl and extract content.
    Format is similar to scifact.
    """
    file_path = os.path.join(nfcorpus_dir, 'nfcorpus_dev.jsonl')
    if not os.path.exists(file_path):
        print(f"NFCorpus data file not found at {file_path}")
        return []

    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {file_path}"):
            data = json.loads(line)
            messages = data.get('messages', [])
            content_pieces = []
            for message in messages:
                message_content = message.get('content', '').strip()
                if message_content:
                    content_pieces.append(message_content)
            content = " ".join(content_pieces)
            if content.strip():
                data_list.append(content)
    return data_list

def load_scifact_data(scifact_dir):
    """
    Load the SciFact dataset from scifact_dev.jsonl and extract content.
    Format is similar to the provided scifact example.
    """
    file_path = os.path.join(scifact_dir, 'scifact_dev.jsonl')
    if not os.path.exists(file_path):
        print(f"SciFact data file not found at {file_path}")
        return []

    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {file_path}"):
            data = json.loads(line)
            messages = data.get('messages', [])
            content_pieces = []
            for message in messages:
                message_content = message.get('content', '').strip()
                if message_content:
                    content_pieces.append(message_content)
            content = " ".join(content_pieces)
            if content.strip():
                data_list.append(content)
    return data_list

def load_vihealthqa_data(vihealthqa_dir):
    """
    Load the vihealthqa dataset from vihealthqa_dev.jsonl and extract content.
    Format is similar to scifact.
    """
    file_path = os.path.join(vihealthqa_dir, 'vihealthqa_dev.jsonl')
    if not os.path.exists(file_path):
        print(f"ViHealthQA data file not found at {file_path}")
        return []

    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {file_path}"):
            data = json.loads(line)
            messages = data.get('messages', [])
            content_pieces = []
            for message in messages:
                message_content = message.get('content', '').strip()
                if message_content:
                    content_pieces.append(message_content)
            content = " ".join(content_pieces)
            if content.strip():
                data_list.append(content)
    return data_list

##################################


# Generate random scores with the same shape and reduction as BM25
for target_task_name in args.target_task_names:
    for train_file_name in args.train_file_names:
        print(f"Processing random scores for training file {train_file_name} and target task {target_task_name}")

        # Load validation data the same way as matching_bm25.py
        validation_path = os.path.join(args.eval_data_path, target_task_name)
        if target_task_name == 'mmlu':
            validation_data = load_mmlu_data(validation_path)
        elif target_task_name == 'bbh':
            validation_data = load_bbh_data(validation_path)
        elif target_task_name == 'tydiqa':
            validation_data = load_tydiqa_data(validation_path)
        elif target_task_name == 'fiqa':
            validation_data = load_fiqa_data(validation_path)
        elif target_task_name == 'nfcorpus':
            validation_data = load_nfcorpus_data(validation_path)
        elif target_task_name == 'scifact':
            validation_data = load_scifact_data(validation_path)
        elif target_task_name == 'vihealthqa':
            validation_data = load_vihealthqa_data(validation_path)
        else:
            if os.path.isdir(validation_path):
                validation_data = []
                for root, dirs, files in os.walk(validation_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.endswith('.jsonl'):
                            validation_data.extend(load_jsonl(file_path))
                        elif file.endswith('.json'):
                            validation_data.extend(load_json(file_path))
            else:
                if validation_path.endswith('.jsonl'):
                    validation_data = load_jsonl(validation_path)
                elif validation_path.endswith('.json'):
                    validation_data = load_json(validation_path)
                else:
                    print(f"Unsupported file format for validation data: {validation_path}")
                    continue

        validation_data = [doc for doc in validation_data if doc.strip()]
        print(f"Loaded {len(validation_data)} validation documents.")

        training_data = load_training_data(train_file_name)
        num_training_samples = len(training_data)
        print(f"Loaded {num_training_samples} training documents.")

        num_validation_samples = len(validation_data)
        n_subtasks = N_SUBTASKS.get(target_task_name, 1)

        if num_validation_samples % n_subtasks != 0:
            print(f"Error: Number of validation samples ({num_validation_samples}) not divisible by {n_subtasks}.")
            continue

        # Create a random influence_score similar to what BM25 does before reduction:
        # BM25 initially creates a [num_validation_samples, num_training_samples] score matrix.
        influence_score = torch.rand((num_validation_samples, num_training_samples), dtype=torch.float32)

        # Apply the same reshaping and reduction steps as in matching_bm25.py:
        num_validation_samples_per_subtask = num_validation_samples // n_subtasks
        influence_score = influence_score.reshape(
            n_subtasks, num_validation_samples_per_subtask, num_training_samples
        ).mean(1).max(0)[0]

        output_dir = os.path.join(args.output_path, target_task_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = os.path.join(output_dir, f"{train_file_name}_random_influence_score.pt")
        torch.save(influence_score, output_file)
        print(f"Saved random influence score to {output_file}")
