import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import logging

# 读取预处理过的数据集
df = pd.read_csv("data.csv")

# 1. 聚合同一IP的所有请求，构建每个IP的所有用户名和密码对的文本
ip_data = {}

for _, row in df.iterrows():
    ip = row['ip']
    username = row['username']
    password = row['password']
    label = row['label']
    
    # 如果该IP没有出现过，初始化
    if ip not in ip_data:
        ip_data[ip] = {
            "usernames_passwords": [],
            "label": label  # 该IP的所有请求的label是一致的
        }
    
    # 将用户名和密码对合并为一个字符串
    ip_data[ip]["usernames_passwords"].append(f"{username} {password}")

# 2. 将每个IP的请求（用户名和密码）拼接成一个序列，进行Tokenization
all_usernames_passwords = []
labels = []

for ip, data in ip_data.items():
    # 合并该IP的所有请求
    all_usernames_passwords.append(" ".join(data["usernames_passwords"]))
    labels.append(data["label"])

#构造DataFrame并保存到CSV文件
data_to_save = {
    'usernames_passwords': all_usernames_passwords,
    'label': labels
}

df_to_save = pd.DataFrame(data_to_save)

# 保存到当前目录下的文件，文件名为 `usernames_passwords_labels.csv`
df_to_save.to_csv('usernames_passwords_labels.csv', index=False)

print("数据已保存到 'usernames_passwords_labels.csv' 文件中。")
# Tokenizer的字符级处理
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(all_usernames_passwords)

# 转换为序列
X = tokenizer.texts_to_sequences(all_usernames_passwords)

# Padding
max_length = max([len(seq) for seq in X])
X = pad_sequences(X, maxlen=max_length)

# 标签
y = np.array(labels)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=False, random_state=42)

# 4. 构建深度学习模型
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=X_train.shape[1]))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification (normal or malicious)

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# 5. 训练模型
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# 6. 评估模型
score = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {score[1]:.4f}")


texts_test = all_usernames_passwords[len(X_train):]  # 获取测试集对应的原始文本

random_index = np.random.choice(len(X_test))  # 随机选择一个测试集的索引
sample_data = X_test[random_index]
true_label = y_test[random_index]
original_text = texts_test[random_index]  # 获取该样本的原始文本数据

# 使用模型进行预测
prediction = model.predict(np.array([sample_data]))  # 预测单条数据

# 获取预测的概率值
prediction_prob = prediction[0][0]

# 输出真实标签、预测标签、原始文本和预测概率
print(f"原始文本：{original_text}")
print(f"真实标签：{'恶意' if true_label == 1 else '正常'}")
print(f"预测结果：{'恶意' if prediction_prob >= 0.5 else '正常'}")
print(f"预测概率：{prediction_prob:.4f}")
