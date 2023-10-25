import os
import importlib.util

dir_path = os.path.dirname(os.path.realpath(__file__))
configuration_file_to_consider = os.path.join(dir_path, "my_conf.py")


def load_module_from_file(filepath):
    spec = importlib.util.spec_from_file_location("conf", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


config = load_module_from_file(configuration_file_to_consider)
logging_folder = config.logging_folder
training_dataset = config.training_dataset
validation_dataset = config.validation_dataset
base_model_id = config.base_model_id
learned_models = config.learned_models
lora_alpha = config.lora_alpha
lora_r = config.lora_r
tokenizer_max_length = config.tokenizer_max_length

