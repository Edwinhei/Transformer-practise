import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# 1. 定义翻译数据集类
class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=32):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_text = f"translate English to Chinese: {self.dataset[index]['translation']['en']}"
        tgt_text = self.dataset[index]['translation']['zh']
        
        return src_text, tgt_text
        # src_encoding = self.tokenizer(
        #     src_text,
        #     max_length=self.max_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt"
        # )
        
        # tgt_encoding = self.tokenizer(
        #     tgt_text,
        #     max_length=self.max_length,
        #     padding="max_length",
        #     truncation=True,
        #     return_tensors="pt"
        # )
        
        # labels = tgt_encoding["input_ids"].squeeze()
        # labels[labels == self.tokenizer.pad_token_id] = -100
        
        # assert src_encoding["input_ids"].max() < self.tokenizer.vocab_size, f"src input_ids out of range: {src_encoding['input_ids'].max()}"
        # assert labels.max() < self.tokenizer.vocab_size or labels.max() == -100, f"labels out of range: {labels.max()}"
        
        # return {
        #     "input_ids": src_encoding["input_ids"].squeeze(),
        #     "attention_mask": src_encoding["attention_mask"].squeeze(),
        #     "labels": labels
        # }

if __name__ == "__main__": 
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    dataset = load_dataset("wmt19", "zh-en")
    
    train_dataset = dataset["train"].select(range(1000))
    
    train_dataset = TranslationDataset(train_dataset, tokenizer, max_length=32)
    # train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    print(train_dataset[2])