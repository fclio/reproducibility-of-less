import json
import os

import torch
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# Helper function to compute rankings
def get_response(model, tokenizer, prompt):
    """
    Generates a response from the model based on the provided prompt.
    """
    # Simulate a conversation
    history = [f"User: {prompt}", "Assistant:"]

    chat_history = "\n".join(history)
    stop_token = tokenizer.eos_token  # Typically, '<|endoftext|>' for GPT-like models

    inputs = tokenizer(chat_history, return_tensors="pt").to(model.device)

    # Generate a response with a stop condition
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=300,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,  # For models that need padding
        eos_token_id=tokenizer.convert_tokens_to_ids(stop_token),  # Stopping at EOS
    )

    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = response.split("Assistant:")[
        -1
    ].strip()  # Extract assistant response

    print(f"Assistant: {response_text}")

    return response_text


def extract_ranking_from_response(response_text):
    """
    Extracts a ranking from the model's response. Assumes the response contains a valid ranking
    formatted as 'X > Y > Z', where X, Y, Z are integers or identifiers.
    """
    try:
        # Attempt to parse a ranking from the response text
        ranking = [int(item.strip("[]")) for item in response_text.split(" > ")]
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
            relevance.append(
                len(true_ranking_ids) - true_ranking_ids.index(pid)
            )  # Rank of the prediction
        else:
            relevance.append(0)  # Assign 0 relevance for missing predictions
    return relevance


def extract_ranking_from_true_ranking(true_ranking, dataset_id):
    """
    Extracts the ranking IDs from the true ranking string.
    Ensures that only non-empty substrings are processed.
    """
    try:
        # Split the ranking string by ' > ', filter out empty strings, and convert to integers
        true_ranking_ids = [
            int(x.strip("[]")) for x in true_ranking.split(" > ") if x.strip()
        ]
        return true_ranking_ids
    except ValueError:
        print(f"Error parsing true ranking: {true_ranking} at {dataset_id}")
        return []


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

    # Add a progress bar using tqdm
    print("Starting evaluation...")
    for example in tqdm(evaluation_data, desc="Processing examples", unit="example"):

        dataset_id = example["id"]
        prompt = example["messages"][0]["content"]
        true_ranking = example["messages"][1]["content"]

        true_ranking_ids = extract_ranking_from_true_ranking(true_ranking, dataset_id)

        response_text = get_response(model, tokenizer, prompt)
        predicted_ranking_ids = extract_ranking_from_response(response_text)
        # Initialize default values for metrics
        ndcg = 0
        mrr = 0

        if predicted_ranking_ids and true_ranking_ids:
            # Compute relevance scores (1 for first place, 2 for second, etc.)
            relevance = calculate_relevance(predicted_ranking_ids, true_ranking_ids)

            # Compute nDCG
            ideal_relevance = sorted(relevance, reverse=True)
            ndcg = ndcg_score([ideal_relevance], [relevance])
            ndcg_scores.append(ndcg)

            # Compute MRR
            first_relevant_index = next(
                (
                    i
                    for i, pid in enumerate(predicted_ranking_ids)
                    if pid in true_ranking_ids
                ),
                None,
            )
            if first_relevant_index is not None:
                # Reciprocal rank (1-based)
                mrr = 1 / (first_relevant_index + 1)
            else:
                mrr = 0  # If no relevant prediction, assign MRR = 0
            mrr_scores.append(mrr)
        else:
            non_response.append(dataset_id)

        # Log results
        results.append(
            {
                "dataset_id": dataset_id,
                "query": prompt,
                "answer": true_ranking,
                "true_ranking": true_ranking_ids,
                "predicted_ranking": predicted_ranking_ids,
                "model_response": response_text,
                "ndcg": ndcg,
                "mrr": mrr,
            }
        )

    # Save metrics as JSON
    metrics = {
        "average_ndcg": sum(ndcg_scores) / len(ndcg_scores),
        "average_mrr": sum(mrr_scores) / len(mrr_scores),
        "number of non_response": len(non_response),
        "details": results,
    }

    dir_name = os.path.dirname(output_path)
    if not os.path.exists(dir_name):
        print(f"Directory {dir_name} does not exist. Creating it...")
        os.makedirs(dir_name)

    print(f"Saving metrics to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation completed. Results saved to {output_path}")
    print(
        f"average_ndcg: {sum(ndcg_scores) / len(ndcg_scores)} ,average_mrr: {sum(
            mrr_scores) / len(mrr_scores)},number of non_response: {len(non_response)}"
    )


# Main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TPDM Evaluation Script with Model Responses"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the TPDM model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the evaluation dataset in JSONL format",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save evaluation results as JSON",
    )

    args = parser.parse_args()

    evaluate_tpdm_with_responses(args.model_path, args.dataset_path, args.output_path)
