import pandas as pd
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms
import librosa
import torch

# 加载电影数据
movies = pd.read_csv("data/movies.csv")

# 初始化 BERT 分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 文本处理函数
def process_text(description):
    tokens = tokenizer(description, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    return tokens['input_ids'], tokens['attention_mask']

# 图像处理函数
def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)

# 音频处理函数
def process_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(audio, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return torch.tensor(mel_spec_db)

# 处理所有数据
processed_data = []
for _, row in movies.iterrows():
    text_tokens = process_text(row['description'])
    image_tensor = process_image(f"data/{row['poster_path']}")
    audio_tensor = process_audio(f"data/{row['audio_path']}")
    processed_data.append((text_tokens, image_tensor, audio_tensor, row['rating']))

# 保存处理后的数据
torch.save(processed_data, "data/processed_data.pt")
print("Data preprocessing completed and saved to 'data/processed_data.pt'")
