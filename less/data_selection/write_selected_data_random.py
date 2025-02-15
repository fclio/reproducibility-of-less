import argparse
import os

import torch


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Script for selecting data for training using random scores"
    )
    argparser.add_argument(
        "--train_file_names",
        type=str,
        nargs="+",
        help="The names of the training score files (without extension)",
    )
    argparser.add_argument(
        "--train_files",
        type=str,
        nargs="+",
        help="The paths of the training files corresponding to each score file",
    )
    argparser.add_argument(
        "--target_task_names",
        type=str,
        nargs="+",
        help="The name(s) of the target tasks",
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        default="selected_data",
        help="The path to the output directory",
    )
    argparser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="The maximum number of samples to select",
    )
    argparser.add_argument(
        "--percentage",
        type=float,
        default=None,
        help="The percentage of the data to be selected",
    )

    args = argparser.parse_args()
    return args


def count_lines(filename):
    with open(filename, "r", encoding="utf-8", errors="ignore") as file:
        return sum(1 for _ in file)


if __name__ == "__main__":
    args = parse_args()
    assert len(args.train_file_names) == len(args.train_files)
    assert args.percentage is not None or args.max_samples is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train_files = len(args.train_file_names)

    for target_task in args.target_task_names:
        output_path = os.path.join(args.output_path, target_task)

        score_paths = [
            os.path.join(output_path, f"{task_name}_random_influence_score.pt")
            for task_name in args.train_file_names
        ]

        num_samples = []
        for score_path in score_paths:
            score = torch.load(score_path, map_location=device)
            num_samples.append(len(score))

        total_samples = sum(num_samples)
        if args.percentage is not None:
            args.max_samples = int(args.percentage * total_samples)
            data_amount_name = f"p{args.percentage}"
        else:
            data_amount_name = f"num{args.max_samples}"

        all_scores = []
        for score_path in score_paths:
            score = torch.load(score_path, map_location=device)
            all_scores.append(score)
        all_scores = torch.cat(all_scores, dim=0)

        # Create indexing tensors
        file_specific_index = torch.cat(
            [torch.arange(line_num) for line_num in num_samples]
        ).to(device)
        data_from = torch.cat(
            [
                torch.ones(line_num, dtype=torch.long) * i
                for i, line_num in enumerate(num_samples)
            ]
        ).to(device)

        # Sort the scores (descending)
        sorted_scores, sorted_idx = torch.sort(all_scores, descending=True)
        data_from = data_from[sorted_idx]
        sorted_index = file_specific_index[sorted_idx]

        # Save sorted indices with scores
        sorted_score_file = os.path.join(output_path, "random_sorted.csv")
        if not os.path.exists(sorted_score_file):
            with open(sorted_score_file, "w", encoding="utf-8") as file:
                file.write("file name, index, score\n")
                for score_val, idx_val, from_val in zip(
                    sorted_scores, sorted_index, data_from
                ):
                    file.write(
                        f"{args.train_file_names[from_val.item()]}, {idx_val.item()}, {
                            round(score_val.item(), 6)}\n"
                    )

        # Read training data lines
        all_lines = []
        for i, train_file in enumerate(args.train_files):
            with open(train_file, "r", encoding="utf-8", errors="ignore") as file:
                lines = file.readlines()
                # Ensure we only take as many lines as we counted before
                lines = lines[: num_samples[i]]
                all_lines.append(lines)

        # Select top samples
        final_index_list = sorted_index[: args.max_samples].tolist()
        final_data_from_list = data_from[: args.max_samples].tolist()

        print(f"Total samples: {total_samples}")
        print(f"num_samples per file: {num_samples}")
        print(f"args.max_samples: {args.max_samples}")
        print(f"Length of final_index_list: {len(final_index_list)}")
        print(f"Length of final_data_from: {len(final_data_from_list)}")

        for i, lines in enumerate(all_lines):
            print(f"File {args.train_files[i]} has {len(lines)} lines.")

        # Write top selected samples
        top_file = os.path.join(
            output_path,
            f"random_top_{
                data_amount_name}.jsonl",
        )
        with open(top_file, "w", encoding="utf-8", errors="ignore") as file:
            for index, data_from_val in zip(final_index_list, final_data_from_list):
                file.write(all_lines[data_from_val][index])

        print(
            f"Saved top {
                args.max_samples} random selected data to {top_file}"
        )
