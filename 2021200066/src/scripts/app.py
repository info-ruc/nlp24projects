import re
import os
import math
import nltk
import time
import jieba
import torch
import pickle
import zipfile
import warnings
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch.nn.functional as F
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertTokenizer, BertForQuestionAnswering, BertModel
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")
#nltk.download('punkt_tab')
@st.cache_resource
def load_models_and_data():
    model_predict1 = BertForSequenceClassification.from_pretrained('D://hotel_review_model_1211161')
    tokenizer_predict1 = AutoTokenizer.from_pretrained('D://hotel_review_model_1211161')

    tokenizer_predict2 = AutoTokenizer.from_pretrained("D://mac_answer")
    model_predict2 = BertModel.from_pretrained("D://mac_answer")

    tokenizer_sen = AutoTokenizer.from_pretrained("D://mac_1214")
    model_sen = BertForSequenceClassification.from_pretrained("D://mac_1214")

    with open("D://model/comment_embeddings_mac01.pkl", "rb") as f:
        ce_01 = pickle.load(f)
    with open("D://model/comment_embeddings_mac02.pkl", "rb") as f:
        ce_02 = pickle.load(f)
    with open("D://model/comment_embeddings_mac03.pkl", "rb") as f:
        ce_03 = pickle.load(f)
    with open("D://model/comment_embeddings_mac12.pkl", "rb") as f:
        ce_12 = pickle.load(f)
    with open("D://model/comment_embeddings_mac13.pkl", "rb") as f:
        ce_13 = pickle.load(f)
    with open("D://model/comment_embeddings_mac23.pkl", "rb") as f:
        ce_23 = pickle.load(f)

    cd_01 = pd.read_parquet(r'D://xiecheng/all_01_1215.parquet')
    cd_02 = pd.read_parquet(r'D://xiecheng/all_02_1215.parquet')
    cd_03 = pd.read_parquet(r'D://xiecheng/all_03_1215.parquet')
    cd_12 = pd.read_parquet(r'D://xiecheng/all_12_1215.parquet')
    cd_13 = pd.read_parquet(r'D://xiecheng/all_13_1215.parquet')
    cd_23 = pd.read_parquet(r'D://xiecheng/all_23_1215.parquet')
    combined_df = pd.read_parquet(r'D://xiecheng/all_1215.parquet')

    return model_predict1, tokenizer_predict1, model_predict2, tokenizer_predict2, model_sen, tokenizer_sen, ce_01, ce_02, ce_03, ce_12, ce_13, ce_23, cd_01, cd_02, cd_03, cd_12, cd_13, cd_23, combined_df

model_predict1, tokenizer_predict1, model_predict2, tokenizer_predict2, model_sen, tokenizer_sen, ce_01, ce_02, ce_03, ce_12, ce_13, ce_23, cd_01, cd_02, cd_03, cd_12, cd_13, cd_23, combined_df= load_models_and_data()

def predict_model_1(target):
    test_review = [target]
    test_encoding = encode_comments(test_review, tokenizer_predict1)
    with torch.no_grad():
        outputs = model_predict1(**test_encoding)
        logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    top_values, top_indices = torch.topk(probabilities, k=2, dim=1)
    index1, index2 = top_indices[0][0].item(), top_indices[0][1].item()
    #print(probabilities)
    predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label, index1, index2

def get_embeddings(sentences, tokenizer, model, max_length=128):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def encode_comments(comments, tokenizer):
    return  tokenizer(list(comments), padding=True, truncation=True, return_tensors="pt", max_length=128)

def analyze_sentiment(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        sentiment = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, sentiment].item()
    return sentiment

# 情感分析函数
def sentiment(query, tokenizer, model, device):
    sentiment = analyze_sentiment(query, tokenizer, model, device)
    sentiment_label = "正向" if sentiment == 1 else "负向"
    return sentiment


# 加载训练好的模型和分词器
def predict_model_2(query):
    mid, i1, i2= predict_model_1(query)
    if (i1 == 0 and i2 == 1) or (i1 == 1 and i2 == 0) :
        comments_df = cd_01
        comment_embeddings = ce_01
    if (i1 == 0 and i2 == 2) or (i1 == 2 and i2 == 0) :
        comments_df = cd_02
        comment_embeddings = ce_02
    if (i1 == 0 and i2 == 3) or (i1 == 3 and i2 == 0) :
        comments_df = cd_03
        comment_embeddings = ce_03
    if (i1 == 2 and i2 == 1) or (i1 == 1 and i2 == 2) :
        comments_df = cd_12
        comment_embeddings = ce_12
    if (i1 == 3 and i2 == 1) or (i1 == 1 and i2 == 3) :
        comments_df = cd_13
        comment_embeddings = ce_13
    if (i1 == 2 and i2 == 3) or (i1 == 3 and i2 == 2) :
        comments_df = cd_23
        comment_embeddings = ce_23

    city = ["北京", "广州", "上海"]
    que_city = ""
    for i in city:
        if i in query:
            que_city = i
    start = time.time()
    comments = comments_df["reviews"].tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_predict2.to(device)
    model_predict2.eval()
    query_embedding = get_embeddings([query], tokenizer_predict2, model_predict2).to(device)
    cosine_scores = cos_sim(query_embedding, comment_embeddings)[0]
    top_k = 30

    top_k_scores, top_k_indices = torch.topk(cosine_scores, k=top_k)
    top_k_data = comments_df.iloc[top_k_indices]
    top_k_comments = [comments[i] for i in top_k_indices]
    model_sen.eval()
    model_sen.to(device)
    end = time.time()
    return top_k_data
    #print(f"时长: {end-start}秒")

def main():
    st.title("酒店推荐小助手 🏨")
    st.write("输入您的需求，我会为您推荐适合的酒店。")

    # 用户输入
    query = st.text_input("请输入您的需求（例如：北京靠近火车站的酒店）", "")
    if st.button("推荐酒店"):
        # 加载数据和模型
        # 推荐结果
        if query.strip():
            results = predict_model_2(query)
            print(results)
            st.write(f"""输入的问题： {query}""")
            st.write("以下是推荐的酒店：")
            printed_hotels = []
            count = 0
            max_count = 5  # 最多输出的酒店数量
            for i, (index, row) in enumerate(results.iterrows()):
                if count >= max_count:
                    break
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sen = sentiment(row['reviews'], tokenizer_sen, model_sen, device)
                if sen != 1:
                    continue
                hotel_name = row['name'] 
                if hotel_name in printed_hotels:
                    continue
                hotel_info = combined_df[combined_df['name'] == hotel_name].iloc[0] 
                city = ["北京", "广州", "上海"]
                que_city = ""
                for i in city:
                    if i in query:
                        que_city = i
                if que_city not in hotel_info['location']:
                    continue
                col1, col2 = st.columns([1, 3])  # 左侧 1/4, 右侧 3/4
                with col1:# 添加图片框
                    image_path = f"D://hotel_image/hotel{count + 1}.jpg"
                    st.image(
                        image_path,  # 假设图片链接存储在 `image_url` 字段
                        caption=f"酒店 {count + 1}",
                        width=300
                    )
                with col2:# 展示酒店信息
                    st.write(f"""酒店 {count + 1}： {hotel_info['name']}""")
                    st.write(f"""地点: {hotel_info['location']}""")
                    st.write(f"""综合评分: {hotel_info['total']}，位置评分: {hotel_info['loc']}，设施评分: {hotel_info['fac']}，服务评分: {hotel_info['ser']}，卫生评分: {hotel_info['hyg']}""")
                    st.write(f"""相关评论: {row['reviews']}。""")
                    st.write("")
                printed_hotels.append(hotel_name)
                count += 1
        #else:
            #st.warning("请输入有效的需求！")

if __name__ == "__main__":
    main()