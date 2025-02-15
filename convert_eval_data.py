import os
import json
import random
import logging
from collections import defaultdict
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from tqdm import tqdm

# Set logging level to suppress BEIR warnings
logging.getLogger("beir").setLevel(logging.ERROR)

def convert_beir_to_pointwise(data_path, num_documents_per_query=10, split="test", dev_mode=False):
    """
    Converts a BEIR dataset into a pointwise format, prioritizing highly relevant documents.

    Args:
        data_path (str): Path to the downloaded BEIR dataset.
        num_documents_per_query (int): Number of documents per query (positive + negative).
        split (str): The dataset split to load (e.g., "test" or "dev").
        dev_mode (bool): If True, ensures each query appears twice (once with a positive, once with a negative).

    Returns:
        list: Reformatted dataset in pointwise format.
    """
    random.seed(42)  # Ensure reproducibility
    pointwise_data = []
    prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."

    # Track query statistics
    query_count = 0
    query_distribution = defaultdict(int)

    # Ensure dataset path exists
    if not os.path.exists(data_path):
        return []

    # Load dataset (corpus = documents, queries = user questions, qrels = relevance labels)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)

    # Store dev queries separately for final selection
    dev_queries = []

    # Process each query in queries.jsonl
    for query_index, (query_id, query_text) in tqdm(enumerate(queries.items(), start=1), desc=f"Processing queries ({split})"):

        query_count += 1  # Track total queries

        # Retrieve relevant documents (prioritizing score `2`)
        highly_relevant_doc_ids = [doc_id for doc_id, score in qrels.get(query_id, {}).items() if score == 2]
        weakly_relevant_doc_ids = [doc_id for doc_id, score in qrels.get(query_id, {}).items() if score == 1]

        # Use only `2`s if possible; otherwise, use both `1`s and `2`s
        positive_doc_ids = highly_relevant_doc_ids if highly_relevant_doc_ids else (highly_relevant_doc_ids + weakly_relevant_doc_ids)
        negative_doc_ids = list(set(corpus.keys()) - set(positive_doc_ids))  # Non-relevant documents

        # Retrieve document texts
        def get_text(doc_id):
            return corpus[doc_id]["text"] if isinstance(corpus[doc_id], dict) else corpus[doc_id]

        positive_passages = [get_text(doc_id) for doc_id in positive_doc_ids if doc_id in corpus]
        negative_passages = [get_text(doc_id) for doc_id in negative_doc_ids if doc_id in corpus]

        # Ensure at least one negative document per query
        if not negative_passages:
            remaining_docs = list(set(corpus.keys()) - set(positive_doc_ids))  # Any doc not marked as positive
            if remaining_docs:
                negative_passages.append(get_text(random.choice(remaining_docs)))

        # Track query distribution
        num_positive = min(len(positive_passages), num_documents_per_query // 2)
        num_negative = num_documents_per_query - num_positive
        query_distribution[(num_positive, num_negative)] += 1

        # Dev Mode: Store queries for final selection (10 queries, each appearing twice)
        if dev_mode:
            if positive_passages and negative_passages:
                dev_queries.append((query_text, positive_passages[0], negative_passages[0]))  # Store first pos & neg

            # Stop after collecting enough queries
            if len(dev_queries) >= 10:
                break
            continue  # Skip normal processing for dev mode

        # Sampling
        sampled_positives = random.sample(positive_passages, num_positive) if num_positive > 0 else []
        sampled_negatives = random.sample(negative_passages, min(num_negative, len(negative_passages)))

        # Assign logical IDs per query
        doc_index = 1

        # Create pointwise examples for positive documents
        for pos in sampled_positives:
            pointwise_data.append({
                "dataset": data_path.split("/")[-1],
                "id": f"{data_path.split('/')[-1]}_{query_index}_{doc_index}",
                "messages": [
                    {"role": "user", "content": f"{prompt} Query: {query_text} Passage: {pos}"},
                    {"role": "assistant", "content": "Yes"}
                ]
            })
            doc_index += 1

        # Create pointwise examples for negative documents
        for neg in sampled_negatives:
            pointwise_data.append({
                "dataset": data_path.split("/")[-1],
                "id": f"{data_path.split('/')[-1]}_{query_index}_{doc_index}",
                "messages": [
                    {"role": "user", "content": f"{prompt} Query: {query_text} Passage: {neg}"},
                    {"role": "assistant", "content": "No"}
                ]
            })
            doc_index += 1

    # For dev mode, ensure exactly 10 queries appear twice (once positive, once negative)
    if dev_mode:
        final_dev_data = []
        for i, (query_text, pos, neg) in enumerate(dev_queries):
            final_dev_data.append({
                "dataset": data_path.split("/")[-1],
                "id": f"{data_path.split('/')[-1]}_dev_{i+1}_pos",
                "messages": [
                    {"role": "user", "content": f"{prompt} Query: {query_text} Passage: {pos}"},
                    {"role": "assistant", "content": "Yes"}
                ]
            })
            final_dev_data.append({
                "dataset": data_path.split("/")[-1],
                "id": f"{data_path.split('/')[-1]}_dev_{i+1}_neg",
                "messages": [
                    {"role": "user", "content": f"{prompt} Query: {query_text} Passage: {neg}"},
                    {"role": "assistant", "content": "No"}
                ]
            })
        return final_dev_data

    # Print query distribution statistics
    print("\n✅ Query Processing Summary:")
    print(f"Total Queries Processed: {query_count}")
    print("Distribution of (Positive-Negative) Documents per Query:")
    for (pos, neg), count in sorted(query_distribution.items(), reverse=True):
        print(f"  {pos}-{neg}: {count} queries")

    return pointwise_data

def process_beir_dataset(subdataset, num_documents_per_query=10, num_dev_queries=10):
    """
    Processes a BEIR dataset and saves both train and dev splits.

    Args:
        subdataset (str): Name of the BEIR dataset (e.g., "nfcorpus", "scidocs").
        num_documents_per_query (int): Number of documents per query in the train set.
        num_dev_queries (int): Number of queries in the dev set (each appears twice).
    """
    print(f"\n🚀 Processing dataset: {subdataset}")

    # Set dataset paths
    data_path = f"./datasets/{subdataset}"
    output_path = f"./data/eval/{subdataset}"

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Download dataset if not present
    if not os.path.exists(data_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{subdataset}.zip"
        util.download_and_unzip(url, os.path.dirname(data_path))

    # Convert test set to pointwise format (Full dataset)
    formatted_data = convert_beir_to_pointwise(data_path, num_documents_per_query, split="test")

    # if subdataset only has test set, sample 10 queries for dev set from test set
    if not os.path.exists(os.path.join(data_path, "dev.jsonl")):
        formatted_dev_data = convert_beir_to_pointwise(data_path, 2, split="test", dev_mode=True)

    else:
        # Convert dev set (10 queries with 1 positive & 1 negative each)
        formatted_dev_data = convert_beir_to_pointwise(data_path, 2, split="dev", dev_mode=True)

    # Save formatted data
    output_file_train = os.path.join(output_path, f"{subdataset}_data.jsonl")
    output_file_dev = os.path.join(output_path, f"{subdataset}_dev.jsonl")

    with open(output_file_train, 'w', encoding='utf-8') as f:
        for entry in formatted_data:
            f.write(json.dumps(entry) + "\n")

    with open(output_file_dev, 'w', encoding='utf-8') as f:
        for entry in formatted_dev_data:
            f.write(json.dumps(entry) + "\n")

    print(f"✅ Full dataset saved to {output_file_train}")
    print(f"✅ 10-query dev subset saved to {output_file_dev}")

# Example usage
datasets = ["nfcorpus", "scidocs", "fiqa", "vihealthqa"]  # Add more datasets as needed
for dataset in datasets:
    process_beir_dataset(dataset)