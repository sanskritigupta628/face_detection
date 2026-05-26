import os
import cv2
import numpy as np
import re
import threading
import uuid
import shutil
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory
from concurrent.futures import ThreadPoolExecutor
from werkzeug.utils import secure_filename
from insightface.app import FaceAnalysis
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)

# Initialize face analysis model
FACE_MODEL_ROOT = "/workspace/FaceDetection"  # Local Models Folder
face_app = FaceAnalysis(
    name='buffalo_l',
    root=FACE_MODEL_ROOT    # This prevents downloading
)
face_app.prepare(ctx_id=0)

# Cache and lock for thread-safe encoding cache access
encoding_cache = {}
encoding_lock = threading.Lock()

LIBRARIES_DIR = "./"

@app.route('/library_image/<library>/<filename>')
@cross_origin()  # Allows all origins by default
def get_library_image(library, filename):
    folder_path = get_library_path(library)
    try:
        return send_from_directory(folder_path, filename)
    except:
        return jsonify({'Status': 'false', 'error': 'Image not found'}), 404

# New route for the UI
@app.route('/ui')
def ui():
    libraries = [d for d in os.listdir(LIBRARIES_DIR) 
                if os.path.isdir(os.path.join(LIBRARIES_DIR, d)) and not d.startswith('.')]
    return render_template('index.html', libraries=libraries)
def sanitize_folder_name(name):
    return re.sub(r'[<>:"/\\|?*]', '', name.strip())


def get_library_path(library_name):
    return os.path.join(LIBRARIES_DIR, sanitize_folder_name(library_name))


def resize_image(img, max_size=640):
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1 else img


def fast_imread(image_path):
    with open(image_path, 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def fast_imdecode(file_bytes):
    return cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)


def process_image(file_path):
    image = fast_imread(file_path)
    if image is None:
        return []
    image = resize_image(image)
    faces = face_app.get(image)
    return [(face.embedding, f"{os.path.basename(file_path)}#face{idx}") for idx, face in enumerate(faces)] if faces else []


def encode_faces(folder_path, force_rebuild=False):
    cache_path = os.path.join(folder_path, 'encodings.npz')
    with encoding_lock:
        if not force_rebuild and folder_path in encoding_cache:
            return encoding_cache[folder_path]

    if not force_rebuild and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        encs, names = data['encodings'], data['file_names'].tolist()
        with encoding_lock:
            encoding_cache[folder_path] = (encs, names)
        return encs, names

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f)) and not f.endswith('.npz')]

    all_encodings, all_file_names = [], []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for face_list in executor.map(process_image, files):
            for emb, fname in face_list:
                all_encodings.append(emb)
                all_file_names.append(fname)

    enc_np = np.array(all_encodings)
    np.savez_compressed(cache_path, encodings=enc_np, file_names=np.array(all_file_names))
    with encoding_lock:
        encoding_cache[folder_path] = (enc_np, all_file_names)

    return enc_np, all_file_names


def update_encodings_with_new_images(folder_path, new_files):
    cache_path = os.path.join(folder_path, 'encodings.npz')
    new_paths = [os.path.join(folder_path, file) for file in new_files]

    new_encodings, new_filenames = [], []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for results in executor.map(process_image, new_paths):
            for emb, fname in results:
                new_encodings.append(emb)
                new_filenames.append(fname)

    if not new_encodings:
        return

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        encodings = list(data['encodings'])
        file_names = list(data['file_names'])
        encodings.extend(new_encodings)
        file_names.extend(new_filenames)
    else:
        encodings = new_encodings
        file_names = new_filenames

    enc_np, fn_np = np.array(encodings), np.array(file_names)
    np.savez_compressed(cache_path, encodings=enc_np, file_names=fn_np)
    with encoding_lock:
        encoding_cache[folder_path] = (enc_np, file_names)


@app.route('/')
def home():
    return jsonify({'Status': 'true', 'message': '✅ Face Recognition API is running!'})


def assess_image_quality(image):
    h, w = image.shape[:2]

    if h < 100 or w < 100:
        return False, "Low resolution", None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 30:
        return False, "Image too blurry", None

    faces = face_app.get(resize_image(image))
    if not faces:
        return False, "No face detected", None

    if blur_score > 100:
        quality = "very_high"
    elif blur_score > 60:
        quality = "high"
    elif blur_score > 30:
        quality = "medium"
    else:
        quality = "low"

    return True, None, quality


@app.route('/upload', methods=['POST'])
def upload_images():
    if 'images' not in request.files or 'library' not in request.form:
        return jsonify({'Status': 'false', 'error': 'Images and library name are required'}), 400

    image_files = request.files.getlist('images')
    library = request.form['library']
    folder_path = get_library_path(library)
    os.makedirs(folder_path, exist_ok=True)

    accepted, rejected = [], []

    for f in image_files:
        if f and f.filename:
            file_bytes = f.read()
            image = fast_imdecode(file_bytes)
            if image is None:
                rejected.append({'filename': f.filename, 'reason': 'Unreadable image'})
                continue

            accepted_flag, reason, quality = assess_image_quality(image)
            if not accepted_flag:
                rejected.append({'filename': f.filename, 'reason': reason})
                continue

            filename = secure_filename(f.filename)
            save_path = os.path.join(folder_path, filename)
            cv2.imwrite(save_path, image)
            accepted.append({'filename': filename, 'quality': quality})

    update_encodings_with_new_images(folder_path, [x['filename'] for x in accepted])

    return jsonify({
        'Status': 'true' if len(accepted) > 0 else 'false',
        'accepted': accepted,
        'rejected': rejected,
        'message': f'✅ {len(accepted)} image(s) accepted, {len(rejected)} rejected.'
    }), 200 if len(accepted) > 0 else 400


@app.route('/single_detect', methods=['POST'])
def compare_api():
    if 'image' not in request.files or 'library' not in request.form:
        return jsonify({'Status': 'false', 'error': 'Image and library name are required'}), 400

    image_file = request.files['image']
    library = request.form['library']
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    image_file.save(temp_filename)

    folder_path = get_library_path(library)
    if not os.path.exists(folder_path):
        os.remove(temp_filename)
        return jsonify({'Status': 'false', 'error': f'Library "{library}" not found'}), 404

    force_rebuild = request.args.get('force_rebuild', 'false').lower() == 'true'
    result = compare_faces(temp_filename, folder_path, force_rebuild=force_rebuild, top_k=5)
    os.remove(temp_filename)

    if isinstance(result, str):
        return jsonify({'Status': 'false', 'message': result}), 404
    else:
        return jsonify({'Status': 'true', 'data': result}), 200


@app.route('/all_detect', methods=['POST'])
def found_faces():
    if 'image' not in request.files or 'library' not in request.form:
        return jsonify({'Status': 'false', 'error': 'Image and library name are required'}), 400

    image_file = request.files['image']
    library = request.form['library']
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    image_file.save(temp_filename)

    folder_path = get_library_path(library)
    if not os.path.exists(folder_path):
        os.remove(temp_filename)
        return jsonify({'Status': 'false', 'error': f'Library "{library}" not found'}), 404

    force_rebuild = request.args.get('force_rebuild', 'false').lower() == 'true'
    result = compare_faces(temp_filename, folder_path, force_rebuild=force_rebuild, top_k=None)
    os.remove(temp_filename)

    if isinstance(result, str):
        return jsonify({'Status': 'false', 'message': result}), 404
    else:
        return jsonify({'Status': 'true', 'data': result}), 200


@app.route('/delete_library', methods=['POST'])
def delete_library():
    if 'library' not in request.form:
        return jsonify({'Status':'false','error': 'Library name is required'}), 400

    library = request.form['library']
    folder_path = get_library_path(library)

    if not os.path.exists(folder_path):
        return jsonify({'Status':'false','error': 'Library not found'}), 404

    shutil.rmtree(folder_path)
    with encoding_lock:
        encoding_cache.pop(folder_path, None)

    return jsonify({'Status':'true','message': f'✅ Library "{library}" deleted successfully.'}), 200


@app.route('/delete_image', methods=['POST'])
def delete_image():
    if 'library' not in request.form or 'filename' not in request.form:
        return jsonify({'Status':'false','error': 'Library and filename are required'}), 400

    library = request.form['library']
    filename = secure_filename(request.form['filename'])
    folder_path = get_library_path(library)
    filepath = os.path.join(folder_path, filename)

    if not os.path.exists(filepath):
        return jsonify({'Status':'false','error': 'Image not found'}), 404

    os.remove(filepath)

    cache_path = os.path.join(folder_path, 'encodings.npz')
    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        encs = list(data['encodings'])
        names = list(data['file_names'])

        filtered = [(e, n) for e, n in zip(encs, names) if not n.startswith(filename + "#")]
        if filtered:
            new_encs, new_names = zip(*filtered)
            np.savez_compressed(cache_path, encodings=np.array(new_encs), file_names=np.array(new_names))
            with encoding_lock:
                encoding_cache[folder_path] = (np.array(new_encs), list(new_names))
        else:
            os.remove(cache_path)
            with encoding_lock:
                encoding_cache.pop(folder_path, None)

    return jsonify({'Status':'true','message': f'✅ Image "{filename}" deleted and encodings updated.'}), 200


def compare_faces(input_image_path, folder_path, threshold=0.5, force_rebuild=False, top_k=5):
    known_encodings, known_files = encode_faces(folder_path, force_rebuild=force_rebuild)
    if len(known_encodings) == 0:
        return "No encodings found in library."

    input_image = fast_imread(input_image_path)
    if input_image is None:
        return "Invalid input image."

    input_image = resize_image(input_image)
    input_faces = face_app.get(input_image)

    if not input_faces:
        return "No face found in input image."

    known_norm = known_encodings / np.linalg.norm(known_encodings, axis=1, keepdims=True)
    matches = []

    for face in input_faces:
        input_embedding = face.embedding / np.linalg.norm(face.embedding)
        similarities = np.dot(known_norm, input_embedding)

        for i, score in enumerate(similarities):
            if score > threshold:
                matched_file, face_index = known_files[i].split('#') if '#' in known_files[i] else (known_files[i], 'face0')
                matches.append({
                    "confidence": float(round(score * 100, 2)),
                    "extension": os.path.splitext(matched_file)[1],
                    "face_index": face_index,
                    "match_file": matched_file
                })

    matches.sort(key=lambda x: x["confidence"], reverse=True)
    if top_k is not None:
        matches = matches[:top_k]

    return {
        "total_matches": len(matches),
        "matches": matches
    }

@app.route('/get_library_images/<library>')
def get_images(library):
    folder_path = get_library_path(library)
    try:
        if not os.path.exists(folder_path):
            return jsonify({'Status': 'false', 'error': 'Library not found'}), 404

        # List image files (filter by extension)
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
        image_files = [f for f in os.listdir(folder_path)
                       if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(image_extensions)]

        return jsonify({'Status': 'true', 'images': image_files})

    except Exception as e:
        return jsonify({'Status': 'false', 'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(LIBRARIES_DIR, exist_ok=True)
    app.run(host='0.0.0.0', port=8003)
