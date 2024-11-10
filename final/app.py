from flask import Flask, request, jsonify, send_file
from PIL import Image
from rembg import remove

app = Flask(__name__)

@app.route('/api/v1/tickets/removeback', methods=['POST'])
def background_remove():
    print(f"Received {request.method} request")
    if 'content_image' not in request.files:
        return jsonify({'error': 'No content image provided'}), 400

    content_file = request.files['content_image']

    # 파일 형식 유효성 검사
    if content_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    try:
        content_image = Image.open(content_file).convert('RGBA')

    except Exception as e:
        return jsonify({'error': 'Invalid image format'}), 400

    # 배경 제거 작업
    try:
        output_image = remove(content_image)

        # 출력 파일 경로 지정 및 저장
        output_path = 'output_image.png'
        output_image.save(output_path, format='PNG')

        # 결과 전송
        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)