import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import ndcg_score
import os
import numpy as np

# Helper function to compute rankings
def get_response(model, tokenizer, prompt, yes_loc):
    """
    Generates a response from the model based on the provided prompt.
    """
    chat_history = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(chat_history, return_tensors="pt").to(model.device)

    outputs = model.forward(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

    # Compute scores for "Yes" token
    scores = (
        outputs.logits[0, 0, yes_loc].item(),  # First occurrence
        outputs.logits[0, :, yes_loc].mean().item()  # Mean score
    )

    return scores

def mean_average_precision(relevance_scores):
    """
    Computes Mean Average Precision (MAP).
    """
    ap_scores = []
    
    for rel in relevance_scores:
        precisions = []
        correct_hits = 0

        for i, val in enumerate(rel):
            if val == 1:
                correct_hits += 1
                precisions.append(correct_hits / (i + 1))

        if precisions:
            ap_scores.append(np.mean(precisions))

    return np.mean(ap_scores) if ap_scores else 0

def precision_at_k(relevance_scores, k):
    """
    Computes Precision@K.
    """
    return np.mean([np.sum(rel[:k]) / k for rel in relevance_scores])

def evaluate_tpdm_with_responses(model_path, dataset_path, output_path):
    """
    Evaluates the TPDM model and calculates ranking-based metrics.
    """
    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Load dataset
    print("Loading evaluation dataset...")
    with open(dataset_path, "r") as f:
        evaluation_data = [json.loads(line) for line in f]

    # Metrics storage
    mrr_scores, ndcg_5_scores, ndcg_10_scores, relevance_lists = [], [], [], []
    precision_5_scores, precision_10_scores = [], []
    non_responses = []

    yes_loc = tokenizer('Yes', return_tensors='pt')['input_ids'][0, 1].item()

    print("Starting evaluation...")
    prompts_per_query = 10
    score_list, relevance_list = [], []
    results = []

    for example in tqdm(evaluation_data, desc="Processing examples", unit="example"):
        dataset_id = example['id']
        prompt = example['messages'][0]['content']
        relevance = example['messages'][1]['content']

        score_0, score_mean = get_response(model, tokenizer, prompt, yes_loc)
        score_list.append(score_0)

        relevance_list.append(1 if relevance == 'Yes' else 0)

        if len(score_list) == prompts_per_query:
            # Sort by model scores
            ranked_indices = np.argsort(score_list)[::-1]
            ranked_relevance = np.array(relevance_list)[ranked_indices]

            # Compute metrics
            ndcg_5 = ndcg_score([relevance_list], [score_list], k=5)
            ndcg_10 = ndcg_score([relevance_list], [score_list], k=10)

            mrr = next((1 / (i + 1) for i, val in enumerate(ranked_relevance) if val == 1), 0)

            relevance_lists.append(ranked_relevance.tolist())
            ndcg_5_scores.append(ndcg_5)
            ndcg_10_scores.append(ndcg_10)
            mrr_scores.append(mrr)

            precision_5_scores.append(np.sum(ranked_relevance[:5]) / 5)
            precision_10_scores.append(np.sum(ranked_relevance[:10]) / 10)

            results.append({
                'dataset_id': dataset_id,
                'query': prompt,
                'ndcg_5': ndcg_5,
                'ndcg_10': ndcg_10,
                'mrr': mrr
            })

            # Reset lists for the next batch
            score_list, relevance_list = [], []

    # Compute MAP
    map_score = mean_average_precision(relevance_lists)

    # Compute final metrics
    metrics = {
        'average_ndcg_5': np.mean(ndcg_5_scores),
        'average_ndcg_10': np.mean(ndcg_10_scores),
        'average_mrr': np.mean(mrr_scores),
        'mean_average_precision': map_score,
        'precision_5': np.mean(precision_5_scores),
        'precision_10': np.mean(precision_10_scores),
        'non_response_count': len(non_responses),
        'details': results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving metrics to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation completed. Results saved to {output_path}")
    print(f"Metrics: {metrics}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TPDM Evaluation Script with Model Responses")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the TPDM model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluation dataset in JSONL format")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save evaluation results as JSON")

    args = parser.parse_args()

    evaluate_tpdm_with_responses(args.model_path, args.dataset_path, args.output_path)
