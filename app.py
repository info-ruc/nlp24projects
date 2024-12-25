from flask import Flask, request, jsonify
import torch

# 加载模型
model = torch.load("models/multimodal_model.pth")
model.eval()

# 创建 Flask 应用
app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    # 输入处理
    text = torch.tensor(data["text"])
    image = torch.tensor(data["image"])
    audio = torch.tensor(data["audio"])
    # 模型预测
    recommendation = model(text, image, audio)
    return jsonify({"recommendation": recommendation.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
