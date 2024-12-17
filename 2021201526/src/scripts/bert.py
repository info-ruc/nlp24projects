# 导入模块
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from transformers import DistilBertModel, DistilBertTokenizer
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("E:\\vscode-L\\PYTHON\\NLP\\data1010.csv")

tokenizer = BertTokenizer.from_pretrained('E:/vscode-L/PYTHON/NLP/bert-base-uncased/')
model = BertModel.from_pretrained('E:/vscode-L/PYTHON/NLP/bert-base-uncased/')

# tokenizer = DistilBertTokenizer.from_pretrained('E:/vscode-L/PYTHON/NLP/distilbert-base-uncased')
# model = DistilBertModel.from_pretrained('E:/vscode-L/PYTHON/NLP/distilbert-base-uncased')

# 将模型设置为评估模式
model.eval()

# 获取 BERT 的文本嵌入
def get_bert_embeddings(texts):
    embeddings = []
    cnt = 0
    for text in texts:
        cnt += 1
        if cnt % 10 == 0:
            print("Process ", cnt, " texts.")

        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        cls_embedding = outputs.last_hidden_state[0, 0, :].numpy()
        embeddings.append(cls_embedding)
    
    return embeddings

embeddings = get_bert_embeddings(df.iloc[:,0].tolist())
embeddings_df = pd.DataFrame(embeddings)

embeddings_df.columns = [f'problemEB_{i}' for i in range(embeddings_df.shape[1])]

# 查看 DataFrame
print(embeddings_df.head())

embeddings_df.to_csv("problem_EB.csv",index=False)