import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import CustomTrainDataset
from input_building import InputBuilder
from model import AsrSpellchecker

model_name = 'rubert-base-cased.bin'

bert_path = 'DeepPavlov/rubert-base-cased'
train_path = '/home/jupyter/datasphere/project/input/train/'
test_path = '/home/jupyter/datasphere/project/input/test/*'
model_path = '../input/coleridgemodels/' + model_name


config = {'tokenizer': AutoTokenizer.from_pretrained(bert_path, do_lower_case=True),
          'bert_model_name': 'DeepPavlov/rubert-base-cased',
          'bert_cfg_path': "bert_config.json",
          'batch_size': 8,
          'Epoch': 1,
          'train_path': train_path,
          'test_path': test_path,
          'device': 'cuda' if torch.cuda.is_available() else 'cpu',
          'max_seq_len': 256,
          'max_spans': 6,
          'model_name': 'asr_spellchecker'
          }

builder = InputBuilder(config)
train_examples, _ = builder.build_input_from_file("train_mini.tsv", False)
dataset = CustomTrainDataset(config['tokenizer'].pad_token_id, train_examples)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=dataset.collate_fn)

test_examples, _ = builder.build_input_from_file("test_mini.tsv", False)
dataset_test = CustomTrainDataset(config['tokenizer'].pad_token_id, test_examples)
dataloader_test = DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=False,
                             collate_fn=dataset_test.collate_fn)

model = AsrSpellchecker(config)
model.run_training(3, dataloader, dataloader_test)
