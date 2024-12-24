import torch
from transformers import BertTokenizer, EncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
from datasets import load_metric

# 加载多语言BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-multilingual-cased', 'bert-base-multilingual-cased')

# 设置特殊标记
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# 模型生成设置
model.config.max_length = 128
model.config.num_beams = 4

# 加载OPUS-100数据集
dataset = load_dataset('opus100', 'en-so')

# 数据增强：回译
def back_translate(dataset, model, tokenizer):
    print("Performing back-translation...")
    back_translations = []
    for example in dataset['train']['translation']:
        # 索马里语到英语翻译
        inputs = tokenizer(example['so'], return_tensors='pt', padding=True, truncation=True)
        outputs = model.generate(inputs['input_ids'], max_length=128)
        english_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 英语到索马里语翻译
        inputs = tokenizer(english_translation, return_tensors='pt', padding=True, truncation=True)
        outputs = model.generate(inputs['input_ids'], max_length=128)
        somali_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

        back_translations.append({'en': english_translation, 'so': somali_translation})
    return back_translations

# 数据增强：引入上下文
def add_context(dataset):
    print("Adding contextual data...")
    contextual_data = []
    for i in range(len(dataset['train']['translation']) - 1):
        current_sentence = dataset['train']['translation'][i]
        next_sentence = dataset['train']['translation'][i + 1]
        contextual_data.append({
            'en': current_sentence['en'] + " " + next_sentence['en'], 
            'so': current_sentence['so'] + " " + next_sentence['so']
        })
    return contextual_data

# 执行回译和上下文增强
back_translation_data = back_translate(dataset, model, tokenizer)
contextual_data = add_context(dataset)

# 合并数据集
augmented_data = dataset['train']['translation'] + back_translation_data + contextual_data
train_data, val_data = train_test_split(augmented_data, test_size=0.2, random_state=42)
train_dataset = Dataset.from_dict({'translation': train_data})
val_dataset = Dataset.from_dict({'translation': val_data})
dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset})

# 数据预处理
def preprocess_function(examples):
    inputs = [ex['en'] for ex in examples['translation']]
    targets = [ex['so'] for ex in examples['translation']]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)

# 定义评估指标
metric = load_metric('sacrebleu')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = [[tokenizer.decode(l, skip_special_tokens=True)] for l in labels]
    result = metric.compute(predictions=decoded_preds, references=labels)
    return {'bleu': result['score']}

# 初始化Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 开始训练
trainer.train()

# 评估模型
results = trainer.evaluate()
print(f"BLEU score: {results['eval_bleu']:.2f}")
