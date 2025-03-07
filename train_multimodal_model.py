import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 加载处理后的数据
processed_data = torch.load("data/processed_data.pt")

# 数据加载器
class MovieDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = MovieDataset(processed_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 多模态模型
class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.text_encoder = nn.Embedding(30522, 768)  # BERT embedding
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 224 * 224, 512)
        )
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 128 * 87, 512)
        )
        self.fc = nn.Linear(768 + 512 + 512, 1)

    def forward(self, text, image, audio):
        text_features = self.text_encoder(text).mean(dim=1)
        image_features = self.image_encoder(image)
        audio_features = self.audio_encoder(audio.unsqueeze(1))
        features = torch.cat((text_features, image_features, audio_features), dim=1)
        return self.fc(features)

# 模型实例化
model = MultimodalModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模型训练
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for text, image, audio, rating in dataloader:
        optimizer.zero_grad()
        outputs = model(text, image, audio)
        loss = criterion(outputs.squeeze(), rating.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "models/multimodal_model.pth")
print("Multimodal model training completed and saved to 'models/multimodal_model.pth'")
