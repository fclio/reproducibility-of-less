import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import ndcg_score
import os
from numpy import argsort

# Helper function to compute rankings
def get_response(model, tokenizer, prompt, yes_loc):
    """
    Generates a response from the model based on the provided prompt.
    """
    # Simulate a conversation
    history = [
        f"User: {prompt}",
        "Assistant:"
    ]

    chat_history = "\n".join(history)
    stop_token = tokenizer.eos_token  # Typically, '<|endoftext|>' for GPT-like models

    inputs = tokenizer(chat_history, return_tensors="pt").to(model.device)

    # Generate a response with a stop condition
    outputs = model.forward(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask
    )
    
    scores = (outputs.logits[0,0,yes_loc].item(), outputs.logits[0,:,yes_loc].mean().item())

    return scores



def extract_ranking_from_response(response_text):
    """
    Extracts a ranking from the model's response. Assumes the response contains a valid ranking
    formatted as 'X > Y > Z', where X, Y, Z are integers or identifiers.
    """
    try:
     
        # Remove square brackets and newline characters
        cleaned_text = response_text.replace('[', '').replace(']', '')
        
        # Split the string based on '>' and strip extra spaces
        ranking = [item.strip() for item in cleaned_text.split('>') if item.strip()]
    
        # ranking = [str(item.strip("[]")) for item in response_text.replace(" ", "").split(">")]
        return ranking
    except ValueError:
        # If parsing fails, return an empty list or a default invalid ranking
        print("Invalid response format, unable to extract ranking.")
        return []

# Helper function to compute relevance
def calculate_relevance(predicted_ranking_ids, true_ranking_ids):
    """
    Computes the relevance scores for the predicted ranking IDs compared to the true ranking IDs.
    Missing predictions are treated as irrelevant with a score of 0.
    """
    relevance = []
    for pid in predicted_ranking_ids:
        if pid in true_ranking_ids:
            relevance.append(len(true_ranking_ids) - true_ranking_ids.index(pid))  # Rank of the prediction
        else:
            relevance.append(0)  # Assign 0 relevance for missing predictions
    return relevance

import re



    
def evaluate_tpdm_with_responses(model_path, dataset_path, output_path):
    # Load the TPDM model and tokenizer
    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    # Load evaluation dataset
    print("Loading evaluation dataset...")
    with open(dataset_path, "r") as f:
        evaluation_data = [json.loads(line) for line in f]

    # Metrics storage
    results = []
    ndcg_scores = []
    mrr_scores = []
    non_response = []

    yes_loc = tokenizer('Yes', return_tensors='pt')['input_ids'][0,1].item()
    print(yes_loc)

    # Add a progress bar using tqdm
    print("Starting evaluation...")
    
    prompts_per_query = 10
    prompt_ctr=0
    score_0_list = []
    score_mean_list = []
    relevance_list = []
    for example in tqdm(evaluation_data, desc="Processing examples", unit="example"):

        dataset_id = example['id']
        prompt = example['messages'][0]['content']
        relevance = example['messages'][1]['content']

        (score_0, score_mean) = get_response(model, tokenizer, prompt, yes_loc)
        score_0_list.append(score_0)
        score_mean_list.append(score_mean)
        if relevance == 'Yes': relevance_list.append(1) 
        else: relevance_list.append(0)
        # predicted_ranking_ids = extract_ranking_from_response(response_text)
        # print(predicted_ranking_ids)

        prompt_ctr += 1
        if prompt_ctr == prompts_per_query:        
            print(score_mean_list, relevance_list)
            # Initialize default values for metrics
            ndcg = 0
            mrr = 0
            try:
            # if predicted_ranking_ids:
                # Compute relevance for predicted ranking
                # relevance_scores = [
                #     true_ranking_ids.get(pid) for pid in predicted_ranking_ids
                # ]

                # Compute the true relevance list for all relevant items in true_ranking_ids
                # true_relevance = [score for score in true_ranking_ids.values()]

                # # Calculate NDCG@5
                # ndcg = ndcg_score(relevance_list, score_0_list)

                # Calculate MRR (Mean Reciprocal Rank)
            
                # for rank, pid in enumerate(predicted_ranking_ids, start=1):
                #     if pid in true_ranking_ids and true_ranking_ids[pid] > 0:
                #         mrr = 1 / rank
                #         break
                for rank in argsort(score_0_list):
                    if relevance_list[rank] == 1:
                        mrr = 1/(rank+1)
                        break

                # Store metrics
                # ndcg_scores.append(ndcg)
                mrr_scores.append(mrr)
            except Exception as e:
                print('yo exception', e)
                non_response.append(dataset_id)

            # Log results
            results.append({
                'dataset_id': dataset_id,
                'query': prompt,
                'ndcg': ndcg,
                'mrr': mrr
            })
            
            score_0_list = []
            score_mean_list = []
            relevance_list = []
            prompt_ctr = 0

    # Save metrics as JSON
    metrics = {
        # 'average_ndcg': sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0,
        'average_mrr': sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
        'number of non_response': len(non_response),
        'details': results
    }
    
    dir_name = os.path.dirname(output_path)
    if not os.path.exists(dir_name):
        print(f"Directory {dir_name} does not exist. Creating it...")
        os.makedirs(dir_name)
        
    print(f"Saving metrics to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation completed. Results saved to {output_path}")
    print(f"average_ndcg: {sum(ndcg_scores) / len(ndcg_scores)} ,average_mrr: {sum(mrr_scores) / len(mrr_scores)},number of non_response: {len(non_response)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TPDM Evaluation Script with Model Responses")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the TPDM model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the evaluation dataset in JSONL format")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save evaluation results as JSON")
    
    args = parser.parse_args()
    
    evaluate_tpdm_with_responses(args.model_path, args.dataset_path, args.output_path)



