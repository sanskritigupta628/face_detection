import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import insightface
from insightface.app import FaceAnalysis

# === Initialize Flask ===
app = Flask(__name__)

# === Initialize InsightFace ===
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use GPUExecutionProvider if you have GPU
face_app.prepare(ctx_id=0)

# === Face Encoding Function ===
def encode_faces(folder_path):
    encodings = []
    file_names = []

    for file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        faces = face_app.get(image)
        if len(faces) == 0:
            continue

        embedding = faces[0].embedding
        encodings.append(embedding)
        file_names.append(file)
    
    return encodings, file_names

# === Face Comparison Function ===
def compare_faces(input_image_path, folder_path, threshold=0.5):
    known_encodings, known_files = encode_faces(folder_path)

    input_image = cv2.imread(input_image_path)
    input_faces = face_app.get(input_image)

    if len(input_faces) == 0:
        return "No face found in input image."

    input_embedding = input_faces[0].embedding

    matches = []

    for i, emb in enumerate(known_encodings):
        similarity = np.dot(input_embedding, emb) / (np.linalg.norm(input_embedding) * np.linalg.norm(emb))
        if similarity > threshold:
            confidence = round(similarity * 100, 2)
            matches.append((known_files[i], confidence))

    matches = sorted(matches, key=lambda x: x[1], reverse=True)
    return matches

# === Flask Route ===
@app.route('/compare', methods=['POST'])
def compare_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Empty file name'}), 400

    # Save uploaded file temporarily
    temp_path = 'temp_input.jpg'
    image_file.save(temp_path)

    # Compare with folder
    folder = 'images'  # Change this if your folder path is different
    result = compare_faces(temp_path, folder)

    # Remove temp file
    os.remove(temp_path)

    if isinstance(result, str):
        return jsonify({'message': result}), 404
    elif result:
        return jsonify({'matches': [{'file': r[0], 'confidence': r[1]} for r in result]})
    else:
        return jsonify({'message': 'No matches found'}), 200

# === Run App ===
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)
