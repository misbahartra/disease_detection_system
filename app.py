from flask import Flask, render_template, request, url_for, send_from_directory
from PIL import Image
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import os
import time
import webview
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import json

app = Flask(__name__)


image = None
gray_image = None

counter = 1
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def calculate_color_moments(image):
    r, g, b = image.split()
    r_mean = np.mean(np.array(r))
    g_mean = np.mean(np.array(g))
    b_mean = np.mean(np.array(b))
    r_std = np.std(np.array(r))
    g_std = np.std(np.array(g))
    b_std = np.std(np.array(b))
    return r_mean, g_mean, b_mean, r_std, g_std, b_std

def calculate_glcm(image):
    # Mengubah gambar menjadi skala abu-abu dan array NumPy yang dapat diubah
    image_gray = np.array(image.convert("L"), dtype=np.uint8)

    # Menghitung GLCM
    glcm = greycomatrix(image_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Menghitung properti GLCM
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]

    # Mengembalikan fitur GLCM dalam bentuk array
    return np.array([contrast, dissimilarity, homogeneity, energy, correlation])

def extract_features(image):
    color_moments = calculate_color_moments(image)
    glcm_features = calculate_glcm(image)
    return np.concatenate([color_moments, glcm_features])


@app.route("/")
def index():
    return render_template("home.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/app")
def apps():
    return render_template("app.html")

@app.route("/service")
def service():
    return render_template("service.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/upload", methods=["POST"])
def upload():
    global image
    image_file = request.files["image"]
    image = Image.open(image_file)

    # Hapus file gambar "test_image" sebelumnya (jika ada)
    test_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test_image.jpg')
    if os.path.exists(test_image_path):
        os.remove(test_image_path)

    # Simpan gambar dengan nama "test_image.jpg"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test_image.jpg')
    image.save(image_path)

    image_url = "/uploads/test_image.jpg"
    return image_url

@app.route("/grayscale", methods=["POST"])
def convert_to_grayscale_route():
    global image, gray_image
    gray_image = image.convert("L")
    filename = "test_gray_image.jpg"

    # Delete previously saved image if it exists
    previous_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(previous_filepath):
        os.remove(previous_filepath)

    gray_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return filename

@app.route("/glcm", methods=["POST"])
def _glcm():
    global gray_image
    gray_image_array = np.array(gray_image)
    glcm = greycomatrix(gray_image_array, [1], [0], symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0][0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0][0]
    energy = greycoprops(glcm, 'energy')[0][0]
    correlation = greycoprops(glcm, 'correlation')[0][0]
    result = f"Nilai Kontras: {contrast}<br>"
    result += f"Homogenitas: {homogeneity}<br>"
    result += f"Energi: {energy}<br>"
    result += f"Korelasi: {correlation}<br>"
    glcm_results = {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation
    }

    return glcm_results


@app.route("/color-moments", methods=["POST"])
def calculate_color_moments_route():
    global image
    image_array = np.array(image)
    r_mean = np.mean(image_array[:, :, 0])
    g_mean = np.mean(image_array[:, :, 1])
    b_mean = np.mean(image_array[:, :, 2])
    r_std = np.std(image_array[:, :, 0])
    g_std = np.std(image_array[:, :, 1])
    b_std = np.std(image_array[:, :, 2])
    result = f"Color Moments:<br>"
    result += f"Rata-rata Red: {r_mean}<br>"
    result += f"Rata-rata Green: {g_mean}<br>"
    result += f"Rata-rata Blue: {b_mean}<br>"
    result += f"Standar Deviasi Red: {r_std}<br>"
    result += f"Standar Deviasi Green: {g_std}<br>"
    result += f"Standar Deviasi Blue: {b_std}<br>"
    color_moment_result = {
        'r_mean': r_mean,
        'g_mean': g_mean,
        'b_mean': b_mean,
        'r_std': r_std,
        'g_std': g_std,
        'b_std': b_std,
    }

    

    return color_moment_result



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def test_model(data, labels):
    print(image)
    # Inisialisasi model SVM
    model = SVC()

    # Pelatihan model dengan fitur Color Moments dan label kelas
    model.fit(data, labels)

    # Contoh pengujian dengan data baru
    test_image_path = "uploads/test_image.jpg"
    test_image = Image.open(test_image_path)
    test_features = extract_features(test_image)

    # Prediksi kelas menggunakan model yang telah dilatih
    predicted_class = model.predict([test_features])

    # Menampilkan hasil prediksi
    print("Prediksi kelas:", predicted_class)

    # Melakukan prediksi pada data uji
    y_pred = model.predict(data)
    # Menghitung akurasi prediksi
    accuracy = accuracy_score(labels, y_pred) * 100
    # Buat dictionary dengan hasil prediksi dan akurasi
    result = {
        "Prediksi Penyakit ": predicted_class.tolist(),
        "Tingkat Akurasi ": accuracy.tolist()
    }

    # Mengembalikan hasil sebagai JSON
    return json.dumps(result)


def train_model(data, labels):
    # Split data menjadi data training dan data uji dengan perbandingan 80:20
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Inisialisasi model SVM
    model = SVC(kernel='linear')

    # Melatih model dengan data training
    model.fit(X_train, y_train)

    # Memprediksi label untuk data uji
    y_pred = model.predict(X_test)

    # Menampilkan hasil evaluasi model
    classification_result = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred) * 100

    # Buat dictionary dengan hasil laporan klasifikasi dan akurasi
    result = {
        "classification_report": classification_result,
        "accuracy": accuracy.tolist()
    }

    # Mengembalikan hasil sebagai JSON
    return json.dumps(result)
    

@app.route("/train", methods=["GET"])
def train_model_route():
    print("Training sedang berjalan")
    dataset_path = 'dataset'
    classes = ['hawar-daun', 'karat-daun', 'bercak-daun', 'sehat']
    data = []
    labels = []

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        image_files = os.listdir(class_path)
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            image = Image.open(image_path)
            features = extract_features(image)
            label = class_name.replace("-", " ").title()
            data.append(features)
            labels.append(label)

    # Ubah data dan label menjadi array NumPy
    data = np.array(data)
    labels = np.array(labels)

    # Panggil fungsi train_model dengan data dan label yang telah dikumpulkan
    train_result = train_model(data, labels)
    test_result = test_model(data, labels)
    result = {
        'train_result': train_result,
        'test_result':test_result

    }
    return result

def create_window():
    webview.create_window("Deteksi Penyakit Daun Jagung", app, width=1500, height=1000, resizable=True)


if __name__ == "__main__":
    from threading import Thread
    thread = Thread(target=create_window)
    thread.start()

    # Jalankan aplikasi Pywebview
    webview.start()

# if __name__ == "__main__":
#     app.run(debug=True)