import json
import random
import os
from datasets import load_dataset
from tqdm import tqdm

def convert_msmarco_to_pointwise(msmarco_data, num_negatives=1):
    """
    Converts MS MARCO dataset into a pointwise format with positive and negative examples.
    
    Args:
        msmarco_data (iterable): Iterable over MS MARCO dataset (queries, passages with labels).
        num_negatives (int): Number of negative passages per query.
    
    Returns:
        list: Reformatted dataset in pointwise format.
    """
    random.seed(42)  # Set seed for reproducibility
    pointwise_data = []
    total_queries = 0
    total_positive = 0
    total_negative = 0
    prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
    
    entry_id = 1  # Start ID from 1
    
    for entry in tqdm(msmarco_data, desc="Processing queries"):
        total_queries += 1
        query = entry["query"]
        
        # Extract passages from dictionary format
        passages_dict = entry.get("passages", {})
        passage_texts = passages_dict.get("passage_text", [])
        is_selected = passages_dict.get("is_selected", [])
        
        if isinstance(passage_texts, list) and isinstance(is_selected, list) and len(passage_texts) == len(is_selected):
            positive_passages = [text for text, selected in zip(passage_texts, is_selected) if selected == 1]
            negative_passages = [text for text, selected in zip(passage_texts, is_selected) if selected == 0]
        else:
            continue
        
        total_positive += len(positive_passages)
        total_negative += len(negative_passages)
        
        # Ensure there is at least one positive passage
        for pos in positive_passages:
            pointwise_data.append({
                "dataset": "msmarco",
                "id": f"msmarco_{entry_id}",
                "messages": [
                    {"role": "user", "content": f"{prompt} Query: {query} Passage: {pos}"},
                    {"role": "assistant", "content": "Yes"}
                ]
            })
            entry_id += 1
        
        # Sample negative passages
        sampled_negatives = random.sample(negative_passages, min(num_negatives, len(negative_passages)))
        for neg in sampled_negatives:
            pointwise_data.append({
                "dataset": "msmarco",
                "id": f"msmarco_{entry_id}",
                "messages": [
                    {"role": "user", "content": f"{prompt} Query: {query} Passage: {neg}"},
                    {"role": "assistant", "content": "No"}
                ]
            })
            entry_id += 1
    
    print("Statistics:")
    print(f"Total queries processed: {total_queries}")
    print(f"Total positive passages: {total_positive}")
    print(f"Total negative passages: {total_negative}")
    return pointwise_data

# Load MS MARCO dataset from Hugging Face
print("Downloading MS MARCO dataset...")
dataset = load_dataset("ms_marco", "v2.1", split="train")

# Convert dataset to pointwise format
formatted_data = convert_msmarco_to_pointwise(dataset)

# Define the relative path
relative_path = 'data/train/processed/msmarco'
os.makedirs(relative_path, exist_ok=True)  # Create the folder if it doesn't exist

# Save the formatted data to a JSONL file
jsonl_path = os.path.join(relative_path, 'msmarco_data.jsonl')
with open(jsonl_path, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, indent=4)
