from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd
import time
import wandb
tqdm.pandas()
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

# # trl-transformer reinforcement learning
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from model import *


model_path = '/data/wenxuan.gao/work2/save_pretrained'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("lvwerra/gpt2-imdb")

#### get a batch from the dataset
bs = 16
game_data = dict()
dataset.set_format("pandas")
df_batch = dataset[:].sample(bs)
game_data['query'] = df_batch['query'].tolist()
query_tensors = df_batch['input_ids'].tolist()

response_tensors_ref, response_tensors = [], []

ppo_trainer = PPOTrainer(config, model, ref_model=ref_model, 
                         tokenizer=tokenizer, dataset=dataset, 
                         data_collator=collator)
#### get response from gpt2 and gpt2_ref
for i in range(bs):
    gen_len = output_length_sampler()
    output = ref_model.generate(torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device),
                                     max_new_tokens=gen_len, **generation_kwargs).squeeze()[-gen_len:]
    response_tensors_ref.append(output) # 

    output = ppo_trainer.generate(torch.tensor(query_tensors[i]).to(device),
                                 max_new_tokens=gen_len, **generation_kwargs ).squeeze()[-gen_len:]
    response_tensors.append(output)

#### decode responses
game_data['response (before)'] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
game_data['response (after)'] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

#### sentiment analysis of query/response pairs before/after
texts = [q + r for q,r in zip(game_data['query'], game_data['response (before)'])]
game_data['rewards (before)'] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

texts = [q + r for q,r in zip(game_data['query'], game_data['response (after)'])]
game_data['rewards (after)'] = [output[1]["score"] for output in sentiment_pipe(texts, **sent_kwargs)]

# store results in a dataframe
df_results = pd.DataFrame(game_data)
print(df_results)

