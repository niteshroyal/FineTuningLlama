# General configuration
# ---------------------
logging_folder = "/home/nitesh/elexir/LlamaFineTuned/logs"
learned_models = "/scratch/c.scmnk4/elexir/LlamaFineTuned/learned_models"

# Configuration for llama_finetuning.llm.py
# -----------------------------------------
# base_model_id = "meta-llama/Llama-2-7b-hf"
# base_model_id = "facebook/opt-350m"
base_model_id = "facebook/bart-base"
lora_r = 32
lora_alpha = 64
tokenizer_max_length = 100

# Configuration for llama_finetuning.load_finetuning_dataset.py
# -------------------------------------------------------------
training_dataset = "/home/nitesh/elexir/LlamaFineTuned/fine_tuning_datasets/indirect_relationships_train.jsonl"
# training_dataset = "/home/nitesh/elexir/LlamaFineTuned/fine_tuning_datasets/notes.jsonl"
# validation_dataset = "/home/nitesh/elexir/LlamaFineTuned/fine_tuning_datasets/notes_validation.jsonl"
validation_dataset = "/home/nitesh/elexir/LlamaFineTuned/fine_tuning_datasets/indirect_relationships_validation.jsonl"
