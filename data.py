from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import default_data_collator
from itertools import chain

def get_dataloader(dataset, tokenizer, batch_size = 1):
    if dataset == 'sciq':
        raw_dataset = load_dataset("sciq")

        tokenize_function = lambda examples: tokenizer(examples['question'])
        tokenized_datasets = raw_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=raw_dataset["train"].column_names,
                    desc="Running tokenizer on dataset",
                )

        lm_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    desc=f"Grouping texts in chunks of 1024",
                )

        train_dataset = lm_datasets["train"]
        eval_dataset = lm_datasets["validation"]

        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size)

        return train_dataloader, eval_dataloader
    
    else: 
        print('Dataset not supported')
        return None

def group_texts(examples, block_size = 1024):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
