from datasets import load_dataset

from conf import configuration

train_dataset = load_dataset('json', data_files=configuration.training_dataset, split='train')
eval_dataset = load_dataset('json', data_files=configuration.validation_dataset, split='train')


def formatting_func(example):
    text = (f'### Question: {example["Question"]}\n'
            f'A: {example["A"]}\n'
            f'B: {example["B"]}\n\n'
            f'### Answer:\n'
            f'{example["Response"]} ### End')
    return text


def formatting_func_for_eval(example):
    text = (f'### Question: {example["Question"]}\n'
            f'A: {example["A"]}\n'
            f'B: {example["B"]}\n\n')
    return text
