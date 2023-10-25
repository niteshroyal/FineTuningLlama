import os
import json
import torch
import logging
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm import LargeLanguageModel

from conf import configuration
from load_finetuning_dataset import formatting_func_for_eval


def initialization():
    log_file = os.path.join(configuration.logging_folder, os.path.splitext(os.path.basename(__file__))[0] + '.log')
    logging.basicConfig(format='%(levelname)s %(asctime)s:  %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', filename=log_file, filemode='w', level=logging.INFO)


class TestLargeLanguageModel(LargeLanguageModel):

    def __init__(self):
        super().__init__()
        self.ft_model = None
        self.init_conf()

    def init_ft_model(self):
        fine_tuned_model_location = os.path.join(os.path.join(configuration.learned_models,
                                                              configuration.base_model_id), "checkpoint-250")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.ft_model = PeftModel.from_pretrained(base_model, fine_tuned_model_location)

    def evaluate(self, eval_prompt):
        model_input = self.tokenizer(eval_prompt, return_tensors="pt").to("cuda")
        self.ft_model.eval()
        with torch.no_grad():
            logging.info(self.tokenizer.decode(self.ft_model.generate(**model_input, max_new_tokens=300,
                                                                      pad_token_id=2)[0], skip_special_tokens=True))


def evaluate_on_validation_dataset(tester):
    with open(configuration.validation_dataset, 'r') as f:
        for line in f:
            datapoint = json.loads(line)
            text = formatting_func_for_eval(datapoint)
            tester.evaluate(text)


if __name__ == '__main__':
    initialization()
    obj = TestLargeLanguageModel()
    obj.init_ft_model()
    evaluate_on_validation_dataset(obj)

