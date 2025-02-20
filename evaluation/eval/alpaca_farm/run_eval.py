import argparse
import json
import logging
import os
import random

import datasets
import torch
import vllm
from eval.utils import (
    dynamic_import_function,
    generate_completions,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
    query_openai_model,
)


def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    logging.info("loading data and model...")
    alpaca_eval_data = datasets.load_dataset(
        "tatsu-lab/alpaca_farm", "alpaca_farm_evaluation"
    )["eval"]
    # alpaca_eval_data = alpaca_eval_data.select(range(2))
    prompts = []
    chat_formatting_function = (
        dynamic_import_function(args.chat_formatting_function)
        if args.use_chat_format
        else None
    )
    for example in alpaca_eval_data:
        prompt = (
            example["instruction"] + "\n\n" + example["input"]
            if example["input"] != ""
            else example["instruction"]
        )
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
        prompts.append(prompt)

    if args.model_name_or_path is not None:
        if args.use_vllm:
            model = vllm.LLM(
                model=args.model_name_or_path,
                tokenizer=(
                    args.tokenizer_name_or_path
                    if args.tokenizer_name_or_path is not None
                    else args.model_name_or_path
                ),
                # tokenizer_mode="slow",
                tensor_parallel_size=torch.cuda.device_count(),
            )
            sampling_params = vllm.SamplingParams(
                temperature=0,  # greedy decoding
                max_tokens=2048,  # maximum we can pass to roberta
            )
            outputs = model.generate(prompts, sampling_params)
            outputs = [it.outputs[0].text for it in outputs]
        else:
            model, tokenizer = load_hf_lm_and_tokenizer(
                model_name_or_path=args.model_name_or_path,
                tokenizer_name_or_path=(
                    args.tokenizer_name_or_path
                    if args.tokenizer_name_or_path is not None
                    else args.model_name_or_path
                ),
                load_in_8bit=args.load_in_8bit,
                device_map=(
                    "balanced_low_0" if torch.cuda.device_count() > 1 else "auto"
                ),
                gptq_model=args.gptq,
                convert_to_half=args.convert_to_half,
                convert_to_bf16=args.convert_to_bf16,
            )
            print(next(model.parameters()).dtype)
            outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0,
                batch_size=args.eval_batch_size if args.eval_batch_size else 1,
            )
    else:
        import openai

        openai.api_key = "7cf72d256d55479383ab6db31cda2fae"
        openai.api_base = "https://pnlpopenai2.openai.azure.com/"
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"  # this may change in the future
        openai_query_cache_path = os.path.join(
            args.save_dir, "openai_query_cache.jsonl"
        )
        openai_func = (
            query_openai_model
            if args.openai_engine == "text-davinci-003"
            else query_openai_chat_model
        )
        results = openai_func(
            engine=args.openai_engine,
            instances=[
                {"id": str(i), "prompt": prompt} for i, prompt in enumerate(prompts)
            ],
            batch_size=args.eval_batch_size if args.eval_batch_size else 10,
            output_path=openai_query_cache_path,
            max_tokens=args.max_new_tokens,
            temperature=0,
            reuse_existing_outputs=True,
        )
        outputs = [result["output"] for result in results]

    model_name = (
        os.path.basename(os.path.normpath(args.model_name_or_path))
        if args.model_name_or_path is not None
        else args.openai_engine
    )
    model_results = []
    for example, output in zip(alpaca_eval_data, outputs):
        example["output"] = output
        example["generator"] = f"{model_name}-greedy-long"
        # fout.write(json.dumps(example) + "\n")
        model_results.append(example)
    with open(
        os.path.join(
            args.save_dir,
            f"{
                model_name}-greedy-long-output.json",
        ),
        "w",
    ) as fout:
        json.dump(model_results, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_path",
        type=str,
        default="data/eval/alpaca_farm/davinci_003_outputs_2048_token.json",
        help="Path to the reference outputs. "
        "Alpaca_eval leaderboard use davinci_003 to generate the reference outputs, "
        "but they limit the max_tokens to 300. Here we regenerated reference outputs with max_tokens=2048.",
    )
    parser.add_argument("--save_dir", type=str, default="results/alpaca_farm")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="If specified, we will use the OpenAI API to generate the predictions.",
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=1, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--convert_to_half",
        action="store_true",
        help="Load model in half.",
    )
    parser.add_argument(
        "--convert_to_bf16",
        action="store_true",
        help="Load model in bf16.",
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model.",
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts.",
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use vLLM to generate the predictions - much faster.",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=512,
        type=int,
        help="Max number of new tokens",
    )
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert (args.model_name_or_path is None) != (
        args.openai_engine is None
    ), "Either model_name_or_path or openai_engine should be specified."
    main(args)
