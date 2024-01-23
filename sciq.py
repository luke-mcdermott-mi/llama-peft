from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
from peft import LoraConfig, get_peft_model
import torch
from utils import * 
from data import get_dataloader
import lm_eval


# |---------------- Model Setup ----------------|
model_name = "TheBloke/Llama-2-7B-fp16"
config = AutoConfig.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name,
        device_map=0,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        config=config)

peft_config = LoraConfig(task_type="text-generation", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)


# |---------------- Train Setup ----------------|
train_dataloader, eval_dataloader = get_dataloader('sciq', tokenizer, batch_size = 1)

for name, param in model.base_model.named_parameters():
    param.requires_grad = 'lora' in name

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, betas=(0.9,0.99), eps=1e-5)
scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
    optimizer,
    num_training_steps=100,
    num_warmup_steps=10,
)
model.train()
model = model.to('cuda:0')

# |---------------- Training ----------------|
lm_eval.tasks.include_path('/home/ubuntu/luke/llama-peft/tasks')
test_acc = evaluate(model, tokenizer) #test this function

print('Starting training...')
for epoch in range(10):
    train_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch['input_ids'] = batch['input_ids'].to('cuda:0')
        batch['labels']= batch['labels'].to('cuda:0')

        outputs = model(**batch)
        loss = outputs.loss
        train_loss += loss.item() / len(batch['labels'])

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    val_loss = 0 #get_loss(eval_dataloader, model) #bug, causing OOM issues even with batch size 1
    test_acc = evaluate(model, tokenizer) #evaluate model with lm_eval

    print('Epoch: ',epoch,', Train Loss: ', train_loss / len(train_dataloader), ', Val Loss: ', val_loss)
