# Reproduction study of LESS: Selecting Influential Data for Targeted Instruction Tuning

This repository contains the code for the reproduction study on the ICML 2024  paper [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/abs/2402.04333). This work proposes a data selection method to select influential data to induce a target capability. This repository is a clone of the [repository by the paper's authors](https://github.com/princeton-nlp/LESS) with the addition of:
- Scripts to reproduce the experiments presented in the original paper ([here is an explanation how to run them](#experiment-script-usage))
- Scripts to perform an ablation study on the number of training checkpoints used in the LESS data selection method
- Support for the LESS method on a top-5 reranking IR task by way of reranking using generative LLMs
- Scripts to perform experiments with the LESS method on said IR task

## Quick install guide

To install the code on the Snellius cluster, run the following command:

```
bash experiment_script.sh install
```

To install the code on a local system (that doesn't use sbatch jobs), use the following command:

```
bash experiment_script.sh install - - bash
```

## Experiment script usage
To reproduce the results presented in the (reproduction study of the) paper, we created an experiment script that can run every stage of the LESS pipeline, as well as installation, the ablation study experiment, the IR extension experiment and evaluation. The script is compatible with local execution as well as execution on the Snellius cluster.

The script has 4 arguments, 3 of which are optional. It is called using:
```
bash experiment_script.sh [pipeline stage] [LLM name] [random seed] [run command] 
```

The `[pipeline stage]` argument denotes what part of LESS is to be executed. More information on this below. The `[LLM name]` is the name of the backbone LLM for the pipeline. The supported LLMs are `llama2-13b` and `llama2-7b`, the latter of which is the default value. The `[random seed]` argument sets the seed number for all random processes in the pipeline (default value is `5`). The `[run command]` argument defines the command used to run the code. The default value is set to `sbatch` for execution on Snellius, but the script can be used for local execution if it's changed to `bash` or any other shell command. The repository and output directories are defined at the top of the script, they are set to work on Snellius by default but can be changed to work with a different system.

The `[pipeline stage]` argument is used to select the stage of the LESS pipeline. Unfortunately it isn't possible to run the whole pipeline in 1 command because some parts of the pipeline may take a few hours to run. To run the default experiment for the LESS method, run the command with `[pipeline_stage]` set to `install`, `warmup`, `datastore`, `matching` `finetune` and `eval` in order and the results will be saved to `experiment_results.txt`. The full list of options for this argument (sorted by pipeline stage) is:

- `install`: runs the installation script
- Stage 1: Warmup training 
  * `warmup`: runs warmup for the default experiment (for the MMLU, TydiQA & BBH tasks)
  * `warmup_ir`: runs warmup for the IR experiment
- Stage 2: Gradient datastore computation
  * `datastore`: creates (train & eval) datastores for all datasets for all checkpoints for the default experiment
  * `datastore_ir`: creates datastores for all checkpoints for all datasets for the IR experiment
- Stage 3: Matching (influence computation & top-k sorting)
  * `matching`: runs LESS data selection on all datastores for the default experiment (separately for MMLU, TydiQA & BBH)
  * `matching_ir`: runs LESS data selection on all datastores for the IR experiment (separately for NFCorpus, SciFact, ViHealthQA & FiQA)
  * `matching_baseline`: selects 5% of data randomly and using BM25 for the default experiment (separately for MMLU, TydiQA & BBH)
  * `matching_baseline_ir`: selects 5% of data randomly and using BM25 for the IR experiment (separately for NFCorpus, SciFact, ViHealthQA & FiQA)
  * `matching_ablation`: runs LESS data selection for each separate checkpoint and various combinations of 2 checkpoints as described in the ablation study
- Stage 4: Model finetuning
  * `finetune`: finetunes model on 5% of data selected by LESS for the default experiment (separately for MMLU, TydiQA & BBH)
  * `finetune_ir`: finetunes model on 5% of data selected by LESS for the IR experiment (separately for NFCorpus, SciFact, ViHealthQA & FiQA)
  * `finetune_transfer`: finetunes llama2-13b on 5% data selected by LESS using llama2-7b for the default experiment
  * `finetune_baseline`: finetunes model on 5% of data selected by baseline methods (random & BM25) for the default experiment
  * `finetune_baseline_ir`: finetunes model on 5% of data selected by baseline methods (random & BM25) for the IR experiment
  * `finetune_ablation`: finetunes model on 5% of data selected by LESS for each separate checkpoint and various combinations of 2 checkpoints as described in the ablation study
- Evaluation
  * `eval`: evaluates the models finetuned on data selected with LESS for the default experiment (separately for MMLU, TydiQA & BBH). Results are appended to `experiment_results.txt`.
  * `eval_ir`: evaluates the models finetuned on data selected with LESS for the IR experiment (separately for NFCorpus, SciFact, ViHealthQA & FiQA). Results are saved in the to `eval/result` directory.
  * `eval_transfer`: evaluates the models finetuned on data selected with LESS-T for the default experiment. Results are appended to `experiment_results.txt`.
  * `eval_baseline`: evaluates the models finetuned on data selected with random selection & BM25 for the default experiment. Results are appended to `experiment_results.txt`.
  * `eval_baseline_ir`: evaluates the models finetuned on data selected with random selection & BM25 for the IR experiment. Results are saved in the to `eval/result` directory.
  * `eval_baseline`: evaluates the models finetuned on data selected with LESS for the ablation study. Results are appended to `experiment_results.txt`.

# The rest of this README file was written by the creators of the original repository.

## Bugs or Questions?
If you have any questions related to the code or the paper, feel free to email Mengzhou (mengzhou@princeton.edu). If you encounter any problems when using the code, or want to report a bug, you can open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
Please cite our paper if you find the repo helpful in your work:

```bibtex
@inproceedings{xia2024less,
   title={{LESS}: Selecting Influential Data for Targeted Instruction Tuning},
   author={Xia, Mengzhou and Malladi, Sadhika and Gururangan, Suchin and Arora, Sanjeev and Chen, Danqi},
   booktitle={International Conference on Machine Learning (ICML)},
   year={2024}
}
```




