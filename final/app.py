from flask import Flask, request, jsonify, send_file
from PIL import Image
from rembg import remove
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import io

app = Flask(__name__)

# =======================
# 1. 배경 제거 API 설정
# =======================

@app.route('/api/v1/tickets/removeback', methods=['POST'])
def background_remove():
    if 'content_image' not in request.files:
        return jsonify({'error': 'No content image provided'}), 400

    content_file = request.files['content_image']

    # 파일 형식 유효성 검사
    if content_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        content_image = Image.open(content_file).convert('RGBA')
    except Exception:
        return jsonify({'error': 'Invalid image format'}), 400

    # 배경 제거 작업
    try:
        output_image = remove(content_image)

        # 이미지를 바이트로 변환하여 클라이언트에게 전송
        img_io = io.BytesIO()
        output_image.save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =======================
# 2. 이미지 생성 API 설정
# =======================

# Generator 모델 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator().to(device)
netG.load_state_dict(torch.load("generator4.pth", map_location=device))
netG.eval()

# 주요 색상 추출 함수
def extract_dominant_colors(image, k=10):
    image = image.resize((64, 64))
    img_array = np.array(image)
    img_array = img_array.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img_array)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

# 색상 대체 함수
def replace_colors(image, generated_colors, poster_colors):
    img_array = np.array(image)
    h, w, _ = img_array.shape
    for i in range(h):
        for j in range(w):
            pixel = img_array[i, j]
            closest_color_idx = np.argmin(np.sum((generated_colors - pixel) ** 2, axis=1))
            img_array[i, j] = poster_colors[closest_color_idx]
    return Image.fromarray(img_array)

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        # 포스터 이미지 파일 받기
        if 'poster' not in request.files:
            return jsonify({"error": "No file part"}), 400

        poster_file = request.files['poster']
        if poster_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        poster_image = Image.open(poster_file)

        # 포스터의 주요 색상 추출
        poster_dominant_colors = extract_dominant_colors(poster_image)

        # 랜덤 노이즈 생성
        noise = torch.randn(1, 100, 1, 1, device=device)

        # 이미지 생성
        with torch.no_grad():  # 평가 시에는 역전파 불필요
            fake_image = netG(noise).detach().cpu()

        # 이미지 변환
        fake_image = fake_image.squeeze(0)
        fake_image = (fake_image + 1) / 2  # 정규화 해제하여 [0, 1] 범위로 변환

        # 텐서를 NumPy 배열로 변환
        fake_image_np = fake_image.numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

        # 생성된 이미지의 주요 색상 추출
        fake_image1 = Image.fromarray((fake_image_np * 255).astype('uint8'))
        generated_dominant_colors = extract_dominant_colors(fake_image1)

        # 생성된 이미지의 색상을 포스터 주요 색상으로 변경
        recolored_image = replace_colors(fake_image1, generated_dominant_colors, poster_dominant_colors)

        # 이미지 크기 확대
        target_size = (256, 256)  # 원하는 크기로 변경 가능
        recolored_image = recolored_image.resize(target_size, Image.NEAREST)

        # 이미지를 바이트로 변환하여 클라이언트에게 전송
        img_io = io.BytesIO()
        recolored_image.save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =======================
# Flask 앱 실행
# =======================
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)