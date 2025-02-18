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

    yes_token_id = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
    no_token_id = tokenizer('No', add_special_tokens=False)['input_ids'][0]

    # # Compute scores for "Yes" token
    # scores = (
    #     outputs.logits[0, 0, yes_loc].item(),  # First occurrence
    #     outputs.logits[0, :, yes_loc].mean().item()  # Mean score
    # )

    yes_token_id = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
    no_token_id = tokenizer('No', add_special_tokens=False)['input_ids'][0]

    print("yes_token_id")
    print(yes_token_id)

    print("no_token_id")
    print(no_token_id)

    # Get the generated token IDs
    generated_ids = outputs.logits.argmax(dim=-1)[0]

    # Find the last non-special token (avoid EOS token if present)
    if generated_ids[-1] == tokenizer.eos_token_id:
        last_token = generated_ids[-2].item()  # Use second-to-last if EOS is last
    else:
        last_token = generated_ids[-1].item()

    # Compute scores for the actual last generated token
    yes_score = outputs.logits[0, -1, yes_token_id].item()  # Logit for 'Yes'
    no_score = outputs.logits[0, -1, no_token_id].item()  # Logit for 'No'

    # Use the last meaningful token for better scoring
    last_token_score = outputs.logits[0, -1, last_token].item()

    # Compute confidence score (difference between "Yes" and "No")
    confidence = yes_score - no_score  # Positive -> "Yes", Negative -> "No"

    print(f"Last generated token ID: {last_token}")
    print(f"Logit for last token: {last_token_score}")
    print(f"Logit for 'Yes': {yes_score}, Logit for 'No': {no_score}")
    print(f"Final prediction: {'Yes' if confidence > 0 else 'No'} (confidence={confidence})")

    scores = (yes_score, no_score)

    return scores


def precision_at_k(relevance_scores, k):
    return np.sum(relevance_scores[:k]) / k

def recall_at_k(relevance_scores, k):
    total_relevant = np.sum(relevance_scores)
    return np.sum(relevance_scores[:k]) / total_relevant if total_relevant > 0 else 0

def f1_at_k(relevance_scores, k):
    prec = precision_at_k(relevance_scores, k)
    rec = recall_at_k(relevance_scores, k)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

def mean_average_precision(relevance_lists):
    ap_scores = []
    for rel in relevance_lists:
        num_relevant = np.sum(rel)
        if num_relevant == 0:
            ap_scores.append(0)
            continue
        
        precision_sum = 0
        count = 0
        for i in range(len(rel)):
            if rel[i] == 1:
                count += 1
                precision_sum += count / (i + 1)
        
        ap_scores.append(precision_sum / num_relevant)
    
    return np.mean(ap_scores)

def discounted_cumulative_gain(relevance_scores, k):
    return np.sum([(rel / np.log2(i + 2)) for i, rel in enumerate(relevance_scores[:k])])

def r_precision(relevance_scores):
    total_relevant = np.sum(relevance_scores)
    return np.sum(relevance_scores[:total_relevant]) / total_relevant if total_relevant > 0 else 0

def fallout_at_k(relevance_scores, k):
    total_irrelevant = len(relevance_scores) - np.sum(relevance_scores)
    false_positives = k - np.sum(relevance_scores[:k])
    return false_positives / total_irrelevant if total_irrelevant > 0 else 0

def success_at_k(relevance_scores, k):
    return 1 if np.sum(relevance_scores[:k]) > 0 else 0

def average_rank_of_relevant(relevance_scores):
    relevant_indices = [i + 1 for i, rel in enumerate(relevance_scores) if rel == 1]
    return np.mean(relevant_indices) if relevant_indices else 0


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
    ndcg_scores = {5: [], 10: [], 50: [], 100: []}
    mrr_scores = []
    precision_scores = {5: [], 10: [], 50: [], 100: []}
    recall_scores = {5: [], 10: [], 50: [], 100: []}
    f1_scores = {5: [], 10: [], 50: [], 100: []}
    r_precision_scores = []
    relevance_lists = []

    yes_loc = tokenizer('Yes', return_tensors='pt')['input_ids'][0, 1].item()

    print("Starting evaluation...")
    prompts_per_query = 10
    score_list, relevance_list = [], []
    results = []

    for example in tqdm(evaluation_data, desc="Processing examples", unit="example"):
        dataset_id = example['id']
        prompt = example['messages'][0]['content']
        relevance = example['messages'][1]['content']

        print(f"Prompt: {prompt}")
        print(f"Relevance: {relevance}")

        score_0, score_mean = get_response(model, tokenizer, prompt, yes_loc)
        score_list.append(score_0)
        relevance_list.append(1 if relevance == 'Yes' else 0)

        if len(score_list) == prompts_per_query:
            # Sort by model scores
            ranked_indices = np.argsort(score_list)[::-1]
            ranked_relevance = np.array(relevance_list)[ranked_indices]

            # Compute metrics
            for k in [5, 10, 50, 100]:
                ndcg_scores[k].append(ndcg_score([relevance_list], [score_list], k=k))
                precision_scores[k].append(precision_at_k(ranked_relevance, k))
                recall_scores[k].append(recall_at_k(ranked_relevance, k))
                f1_scores[k].append(f1_at_k(ranked_relevance, k))
            
            mrr_scores.append(next((1 / (i + 1) for i, val in enumerate(ranked_relevance) if val == 1), 0))
            r_precision_scores.append(r_precision(ranked_relevance))
            relevance_lists.append(ranked_relevance.tolist())

            results.append({
                'dataset_id': dataset_id,
                'ndcg_5': ndcg_scores[5][-1],
                'ndcg_10': ndcg_scores[10][-1],
                'ndcg_50': ndcg_scores[50][-1],
                'ndcg_100': ndcg_scores[100][-1],
                'mrr': mrr_scores[-1],
                'precision_5': precision_scores[5][-1],
                'precision_10': precision_scores[10][-1],
                'precision_50': precision_scores[50][-1],
                'precision_100': precision_scores[100][-1],
                'recall_5': recall_scores[5][-1],
                'recall_10': recall_scores[10][-1],
                'recall_50': recall_scores[50][-1],
                'recall_100': recall_scores[100][-1],
                'f1_5': f1_scores[5][-1],
                'f1_10': f1_scores[10][-1],
                'f1_50': f1_scores[50][-1],
                'f1_100': f1_scores[100][-1],
                'r_precision': r_precision_scores[-1]
            })
            
            # Reset lists for the next batch
            score_list, relevance_list = [], []

    # Compute MAP
    map_score = mean_average_precision(relevance_lists)
    
    metrics = {
        'average_ndcg_5': np.mean(ndcg_scores[5]),
        'average_ndcg_10': np.mean(ndcg_scores[10]),
        'average_ndcg_50': np.mean(ndcg_scores[50]),
        'average_ndcg_100': np.mean(ndcg_scores[100]),
        'average_mrr': np.mean(mrr_scores),
        'mean_average_precision': map_score,
        'average_precision_5': np.mean(precision_scores[5]),
        'average_precision_10': np.mean(precision_scores[10]),
        'average_precision_50': np.mean(precision_scores[50]),
        'average_precision_100': np.mean(precision_scores[100]),
        'average_recall_5': np.mean(recall_scores[5]),
        'average_recall_10': np.mean(recall_scores[10]),
        'average_recall_50': np.mean(recall_scores[50]),
        'average_recall_100': np.mean(recall_scores[100]),
        'average_f1_5': np.mean(f1_scores[5]),
        'average_f1_10': np.mean(f1_scores[10]),
        'average_f1_50': np.mean(f1_scores[50]),
        'average_f1_100': np.mean(f1_scores[100]),
        'average_r_precision': np.mean(r_precision_scores),
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
