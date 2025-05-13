import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sentencepiece

print("SentencePiece安装成功！")

print("环境准备好了！")

# 选择模型
model_name = "t5-small"

# 加载分词器和模型
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

print("T5模型和分词器加载完成！")

# 输入文本
input_text = "translate English to French: Hello, how are you?"

# 编码输入
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# 生成输出
output = model.generate(input_ids)

# 解码输出
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("翻译结果：", output_text)

# 微调T5模型（可选）
from transformers import Trainer, TrainingArguments

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",                 # 输出目录
    num_train_epochs=3,                     # 训练轮数
    per_device_train_batch_size=16,         # 训练批次大小
    per_device_eval_batch_size=64,          # 评估批次大小
    warmup_steps=500,                       # 预热步数
    weight_decay=0.01,                      # 权重衰减
    logging_dir="./logs"                    # 日志目录
)

dataset = {
    "train": "训练数据文件...",
    "validation": "验证数据文件"
}

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args = training_args,
    eval_dataset=dataset["validation"]
)

# 开始训练
trainer.train()