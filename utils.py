from lm_eval import evaluator
import lm_eval
import torch
from lm_eval.models.huggingface import HFLM

def get_loss(dataloader, model):
    model.eval()
    running_loss = 0
    for step, batch in enumerate(dataloader):
        batch['input_ids'] = batch['input_ids'].to('cuda:0')
        batch['labels'] = batch['labels'].to('cuda:0')

        outputs = model(**batch)
        loss = outputs.loss
        running_loss += loss / len(batch['labels'])
    model.train()
    return running_loss / len(dataloader)

def evaluate(model, tokenizer):
    lm_eval_model = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=["sciq"],
        batch_size=1,
    )
    print(results['results']['sciq'])
    return results['results']['sciq']['acc,none']
