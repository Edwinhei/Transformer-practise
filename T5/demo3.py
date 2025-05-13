import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import os

# 设置调试环境变量
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

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
        
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        labels = tgt_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        assert src_encoding["input_ids"].max() < self.tokenizer.vocab_size, f"src input_ids out of range: {src_encoding['input_ids'].max()}"
        assert labels.max() < self.tokenizer.vocab_size or labels.max() == -100, f"labels out of range: {labels.max()}"
        
        return {
            "input_ids": src_encoding["input_ids"].squeeze(),
            "attention_mask": src_encoding["attention_mask"].squeeze(),
            "labels": labels
        }

# 2. 准备数据和模型
def prepare_data_and_model():
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    
    dataset = load_dataset("wmt19", "zh-en")
    train_dataset = dataset["train"].select(range(1000))
    
    train_dataset = TranslationDataset(train_dataset, tokenizer, max_length=32)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    return model, tokenizer, train_dataloader

# 3. 训练函数
def train_model(model, tokenizer, train_dataloader, epochs=3, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 构造 decoder_input_ids
            decoder_input_ids = torch.full_like(labels, tokenizer.pad_token_id)
            decoder_input_ids[:, 1:] = labels[:, :-1]
            
            # 调试：检查所有输入
            print(f"Batch {batch_idx}:")
            print(f"input_ids max: {input_ids.max().item()}, min: {input_ids.min().item()}")
            print(f"decoder_input_ids max: {decoder_input_ids.max().item()}, min: {decoder_input_ids.min().item()}")
            print(f"labels max: {labels.max().item()}, min: {labels.min().item()}")
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    
    return model

# 4. 测试翻译函数
def translate(model, tokenizer, text, max_length=32):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    inputs = tokenizer(
        f"translate English to Chinese: {text}",
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 5. 主函数
def main():
    model, tokenizer, train_dataloader = prepare_data_and_model()
    
    trained_model = train_model(model, tokenizer, train_dataloader, epochs=3)
    
    trained_model.save_pretrained("t5_small_en2zh")
    tokenizer.save_pretrained("t5_small_en2zh")
    
    test_text = "Hello, how are you today?"
    translation = translate(trained_model, tokenizer, test_text)
    
    print(f"Input: {test_text}")
    print(f"Translation: {translation}")

if __name__ == "__main__":
    main()