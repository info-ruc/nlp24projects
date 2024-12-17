import pandas as pd
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, EncoderDecoderCache
import torch
from sklearn.model_selection import train_test_split

# 加载数据集
data = pd.read_csv('input_data.csv')

# 将数据分为训练集和验证集
train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)

# 将数据转为 Hugging Face Dataset 格式
train_dataset = Dataset.from_pandas(train_data)
eval_dataset = Dataset.from_pandas(eval_data)

# 加载T5的tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')
print("load success")

# 定义数据预处理函数
def preprocess_function(examples):
    inputs = ["correct: " + (text if text is not None else "") for text in examples['incorrect_text']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    outputs = [(text if text is not None else "") for text in examples['correct_text']]
    labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length")

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# 对数据集应用预处理
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 加载预训练的T5模型
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_steps=5000,
    logging_steps=500,
    do_train=True,
    do_eval=True,
)

# 创建Trainer对象
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 微调模型
trainer.train()

# 保存训练好的模型
model.save_pretrained('./corrector_model')
tokenizer.save_pretrained('./corrector_model')

# 训练完成后，进行模型推理
def correct_text(text):
    input_text = "correct: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    past_key_values = None
    past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
    # 生成纠正后的文本
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=512, num_beams=4,past_key_values = past_key_values, length_penalty=2.0, early_stopping=True)

    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

# 示例输入
incorrect_text = "I has a apple."
corrected_text = correct_text(incorrect_text)
print(f"Original text: {incorrect_text}")
print(f"Corrected text: {corrected_text}")
