from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import pandas as pd
import time
import wandb
tqdm.pandas()
from datasets import load_dataset

# 使用lora进行微调
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

# trl-transformer reinforcement learning
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# configuration
config = PPOConfig(
    model_name="lvwerra/gpt2-imdb", # Huggingface中利用imdb数据集对gpt2微调的模型 相当于SFT步骤
    learning_rate=1.4e-5, 
    steps=20000,
    batch_size=256,
    forward_batch_size=16,
    ppo_epochs=4,
    init_kl_coef=0.2,
    target=6, # Target KL value for adaptive KL control
    log_with="wandb", # 使用wandb监视训练过程，也可以使用tensorboard
)
#    log_with="tensorboard", # 使用wandb监视训练过程，也可以使用tensorboard
#    accelerator_kwargs={"logging_dir": "./tb_logger"} # 使用tensorboard加上这一行

wandb.init(name='run-imdb', project='gpt2-test', config=config, )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe_device = 0 if torch.cuda.is_available() else -1

# step 1: 加载模型
pretrained_model = AutoModelForCausalLM.from_pretrained(config.model_name)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token # 设置pad_token和eos_token相同

# 设置目标模块名称
target_modules = None
target_modules = ["c_attn"]  # workaround to use 8bit training on this model

# 设置lora配置参数
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=target_modules,  # handled automatically by peft
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# step 2: 设置8bit训练
pretrained_model = prepare_model_for_int8_training(pretrained_model)

# step 3: 设置lora模型。做instruction learning，到这里就好了。如果要做RLHF，还要做第四步。
pretrained_model = get_peft_model(pretrained_model, lora_config)

# step 4: 将lora模型加载入trl模型，加上value head。
model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

# 做必要的设置，梯度检查。
model.gradient_checkpointing_disable = model.pretrained_model.gradient_checkpointing_disable
model.gradient_checkpointing_enable = model.pretrained_model.gradient_checkpointing_enable

# 加载了两次模型：第一次加载的模型用来进行强化学习，调整参数；第二次加载的模型作为参考模型。
# 计算两个模型的KL散度，用来作为PPO训练的额外奖励信号，来保证我们的模型不会太偏离原始模型（即防止灾难性遗忘情况的发生）。
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

wandb.watch(model, log='all') #观察模型

model.to(device)
ref_model.to(device)

# 加载IMDB数据集 
def build_dataset(tokenizer, dataset_name='imdb', input_min_text_length=2, input_max_text_length=8):
    """ 
    Args:
        dataset_name (`str`): 
            数据集名称
    
    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            返回dataloader
    """
    # 加载IMDB数据集，从huggingface的hub上下载数据，当然也可以下载其他数据
    ds = load_dataset(dataset_name, split='train') # 加载后是DataFrame格式
    ds = ds.rename_columns({'text': 'review'})
    ds = ds.filter(lambda x: len(x["review"])>200, batched=False) # filter指len(x["review"])>200都过滤掉

    # 对batch_size进行裁剪，缩小到2到8之间。（2和8是函数中的默认参数）
    # 在tokenize之前，随机截断输入数据作为待续写的prompt，即query的token长度控制在2到8之间
    input_size = LengthSampler(input_min_text_length, input_max_text_length)
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[:input_size()] # 后面设置batched=False,每次input_size都不同
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    # 将数值型变量设置为torch的tensor格式，并且输出所有的列数据，在RL截断需要使用！一定要注意设置output_all_columns=True
    ds.set_format(type='torch', columns=["input_ids", "label"], output_all_columns=True)
    return ds

dataset = build_dataset(tokenizer)
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# load一个pipeline影评分类器
sent_kwargs = {
    "return_all_scores": True, # 文本生成的参数，这里设置为True，表示生成文本时返回得分
    "function_to_apply": "none", 
    "batch_size": config.forward_batch_size 
}

# 加载在IMDB数据集上微调过的BERT分类器来得到拼接后文本的得分
sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=pipe_device)

# 配置PPO强化学习训练对象
ppo_trainer = PPOTrainer(config, model, ref_model=ref_model, 
                         tokenizer=tokenizer, dataset=dataset, 
                         data_collator=collator)

# 根据query生成response，这里的配置使用top_p和随机采样来生成文本。
generation_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
}

if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu" # to avoid a `pipeline` bug

output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)
#output_length_sampler() #4-16随机

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    logs, timing = dict(), dict()
    t0 = time.time()

    query_tensors = batch['input_ids']
    
    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True
    
    #### Get response from gpt2
    t = time.time()
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    timing['time/get_response'] = time.time() - t

    #### Compute sentiment score
    t = time.time()
    texts = [q + r for q,r in zip(batch['query'], batch['response'])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs] 
    # print(rewards)
    #若一个prompt目前是negative,它的positive score是-0.5，那么加到奖励里面，相当于让它少学这个
    timing['time/get_sentiment_preds'] = time.time()-t
    
    model.gradient_checkpointing_enable()
    model.pretrained_model.config.use_cache = False
    
    #### Run PPO step 
    t = time.time()
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    timing['time/PPOoptimization'] = time.time()-t
    
    # ppo_trainer.log_stats(stats, batch, rewards)
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards).to(device)


    #### Log everything
    timing['time/epoch'] = time.time()-t0
    table_rows = [list(r) for r in zip(batch['query'], batch['response'], rewards.cpu().tolist())]
    logs.update({'game_log': wandb.Table(columns=['query', 'response', 'reward'], rows=table_rows)})
    logs.update(timing)
    logs.update(stats)
    logs["env/reward_mean"] = torch.mean(rewards).cpu().numpy().item()
    logs["env/reward_std"] = torch.std(rewards).cpu().numpy().item()
    logs["env/reward_dist"] = rewards.cpu().numpy()
    wandb.log(logs)


model.save_pretrained('/data/wenxuan.gao/work2/save_pretrained')
tokenizer.save_pretrained('/data/wenxuan.gao/work2/save_pretrained')

