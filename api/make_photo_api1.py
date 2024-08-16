import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request, jsonify, send_file
import numpy as np
import PIL.Image as Image
import io

app = Flask(__name__)

# 이미지 전처리 함수
def load_img(img):
    max_dim = 512
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# 텐서를 이미지로 변환하는 함수
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# 이미지를 numpy 배열로 변환하는 함수 (Pillow 사용)
def img_to_array(img):
    return np.array(img)

# TFHub 스타일 전이 모델 로드
style_transfer_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    print(f"Received {request.method} request")
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return jsonify({'error': 'No content or style image provided'}), 400

    content_file = request.files['content_image']
    style_file = request.files['style_image']

    # 파일 형식 유효성 검사
    if content_file.filename == '' or style_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Pillow로 이미지를 열고 numpy 배열로 변환
        content_image = Image.open(content_file)
        style_image = Image.open(style_file)
    except Exception as e:
        return jsonify({'error': 'Invalid image format'}), 400

    # 이미지를 numpy 배열로 변환 후 전처리
    content_image = img_to_array(content_image)
    style_image = img_to_array(style_image)

    content_image = load_img(content_image)
    style_image = load_img(style_image)

    # 스타일 전이 실행
    stylized_image = style_transfer_model(tf.constant(content_image), tf.constant(style_image))[0]
    stylized_image = tensor_to_image(stylized_image)

    # 이미지를 바이트로 변환하여 클라이언트에게 전송
    img_io = io.BytesIO()
    stylized_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

