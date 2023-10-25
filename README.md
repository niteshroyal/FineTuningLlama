# FineTuningLlama

This repository contains code to fine-tune open source LLMs like Llama 2.

## Installation

```commandline
conda create -n llm python=3.10.11
conda activate llm
```

Installing dependencies

```commandline
pip install bitsandbytes
pip install git+https://github.com/huggingface/transformers.git
pip install git+https://github.com/huggingface/peft.git
pip install git+https://github.com/huggingface/accelerate.git
pip install datasets scipy ipywidgets matplotlib
```

## Fine tuning
The following command will finetune on `fine_tuning_datasets/indirect_relationships_train.jsonl` dataset and validate on `fine_tuning_datasets/indirect_relationships_validation.jsonl` dataset.

```commandline
python llama_finetuning/llm.py
```

Make changes to the configuration by editing the configuration file `conf/my_conf.py`.

The format in which the LLM models (default is bart-base) should see the finetuning dataset can be set by making suitable changes to `formatting_func` in `llama_finetuning/load_finetuning_dataset.py` file.

## Evaluation
The following command will evalute the finetuned model on `fine_tuning_datasets/indirect_relationships_validation.jsonl` dataset.
 
```commandline
python llama_finetuning/test_finetuned.py
```
 
The format in which the finetuned model should see the evaluation dataset can be set by making suitable changes to `formatting_func_for_eval` in `llama_finetuning/load_finetuning_dataset.py` file.
