from flask import Flask, render_template, request, url_for, send_from_directory, jsonify
from PIL import Image
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import os
import time
import webview
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)


image = None
gray_image = None

counter = 1
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




# Fungsi untuk menghitung momen warna
def calculate_color_moments(image):
    if image.mode == "RGB":
        r, g, b = image.split()
        r_mean = np.mean(np.array(r))
        g_mean = np.mean(np.array(g))
        b_mean = np.mean(np.array(b))
        r_std = np.std(np.array(r))
        g_std = np.std(np.array(g))
        b_std = np.std(np.array(b))
    elif image.mode == "CMYK":
        # Konversi gambar CMYK ke RGB
        image = image.convert("RGB")
        r, g, b = image.split()
        r_mean = np.mean(np.array(r))
        g_mean = np.mean(np.array(g))
        b_mean = np.mean(np.array(b))
        r_std = np.std(np.array(r))
        g_std = np.std(np.array(g))
        b_std = np.std(np.array(b))
    elif image.mode == "L":
        # Jika gambar dalam skala abu-abu
        gray_image = np.array(image)
        r_mean = np.mean(gray_image)
        g_mean = np.mean(gray_image)
        b_mean = np.mean(gray_image)
        r_std = np.std(gray_image)
        g_std = np.std(gray_image)
        b_std = np.std(gray_image)
    elif image.mode == "RGBA":
        # Konversi gambar RGBA ke RGB
        image = image.convert("RGB")
        r, g, b = image.split()
        r_mean = np.mean(np.array(r))
        g_mean = np.mean(np.array(g))
        b_mean = np.mean(np.array(b))
        r_std = np.std(np.array(r))
        g_std = np.std(np.array(g))
        b_std = np.std(np.array(b))
    else:
        # Format gambar tidak didukung
        print("Format gambar:", image.format)
        print("Nama gambar:", image.filename)
        raise ValueError("Format gambar tidak didukung.")
        
    return r_mean, g_mean, b_mean, r_std, g_std, b_std

# Fungsi untuk menghitung GLCM
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

# Route untuk halaman utama
@app.route("/")
def index():
    return render_template("home.html")

# Route untuk halaman kontak
@app.route("/contact")
def contact():
    return render_template("contact.html")

# Route untuk halaman aplikasi
@app.route("/app")
def apps():
    return render_template("app.html")

# Route untuk halaman service
@app.route("/service")
def service():
    return render_template("service.html")

# Route untuk halaman about
@app.route("/about")
def about():
    return render_template("about.html")

# Route untuk mengunggah gambar
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

# Route untuk mengubah gambar menjadi skala abu-abu
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

# Route untuk menghitung GLCM
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

# Route untuk menghitung momen warna
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


# Route untuk mengakses file yang diunggah
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Fungsi untuk melakukan pengujian model
def test_model(data, labels):
    print(image)
    # Inisialisasi model SVM
    model = SVC()

    # Pelatihan model dengan fitur Color Moments & GLCM dan label kelas
    model.fit(data, labels)

    # Contoh pengujian dengan data baru
    test_image_path = "uploads/test_image.jpg"
    test_image = Image.open(test_image_path)
    test_features = extract_features(test_image)

    # Prediksi kelas menggunakan model yang telah dilatih
    predicted_class = model.predict([test_features])

    # Menampilkan hasil prediksi ke terminal
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


import joblib

@app.route("/train", methods=["GET"])
def train_model_route():
    print("Training sedang berjalan")

    dataset_path = 'dataset'
    classes = ['hawar-daun', 'karat-daun', 'bercak-daun', 'sehat']
    scenarios = [
        {'name': 'scenario_1', 'train_size': 150, 'test_size': 50},
        {'name': 'scenario_2', 'train_size': 170, 'test_size': 30},
        {'name': 'scenario_3', 'train_size': 160, 'test_size': 40}
    ]

    results = []

    for scenario in scenarios:
        scenario_name = scenario['name']
        train_size = scenario['train_size']
        test_size = scenario['test_size']

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        for class_name in classes:
            # Memuat data training
            train_path = os.path.join(dataset_path, scenario_name, 'train', class_name)
            image_files = os.listdir(train_path)
            for image_file in image_files:
                image_path = os.path.join(train_path, image_file)
                image = Image.open(image_path)
                features = extract_features(image)
                label = class_name.replace("-", " ").title()
                train_data.append(features)
                train_labels.append(label)

            # Memuat data test
            test_path = os.path.join(dataset_path, scenario_name, 'test', class_name)
            image_files = os.listdir(test_path)
            for image_file in image_files:
                image_path = os.path.join(test_path, image_file)
                image = Image.open(image_path)
                features = extract_features(image)
                label = class_name.replace("-", " ").title()
                test_data.append(features)
                test_labels.append(label)

        # Ubah data dan label menjadi array NumPy
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        # Inisialisasi model SVM
        model = SVC(kernel='linear')

        # Melatih model dengan data training
        model.fit(train_data, train_labels)

        # Menyimpan model ke file
        model_file = f'model_{scenario_name}.pkl'
        joblib.dump(model, model_file)

        # Melakukan prediksi pada data training
        y_pred_train = model.predict(train_data)

        # Menampilkan hasil evaluasi model pada data training
        classification_result_train = classification_report(train_labels, y_pred_train, output_dict=True)
        accuracy_train = accuracy_score(train_labels, y_pred_train) * 100

        # Melakukan prediksi pada data test
        y_pred_test = model.predict(test_data)

        # Menampilkan hasil evaluasi model pada data test
        classification_result_test = classification_report(test_labels, y_pred_test, output_dict=True)
        accuracy_test = accuracy_score(test_labels, y_pred_test) * 100

        print(f"Hasil Evaluasi pada Skenario {scenario_name}:")
        print("Classification Report (Training):")
        print(classification_result_train)
        print("Accuracy (Training):", accuracy_train)
        print("Classification Report (Test):")
        print(classification_result_test)
        print("Accuracy (Test):", accuracy_test)

        result = {
            'scenario': scenario_name,
            'accuracy_train': accuracy_train,
            'classification_report_train': classification_result_train,
            'accuracy_test': accuracy_test,
            'classification_report_test': classification_result_test,
            'model_file': model_file
        }
        results.append(result)

    return jsonify(results)



from scipy.spatial.distance import euclidean



# Fungsi untuk menghitung persentase kemiripan
def calculate_similarity_percentage(test_data, model):
    class_distances = model.decision_function(test_data)
    max_distance = np.max(class_distances)
    min_distance = np.min(class_distances)
    similarity_percentage = ((max_distance - class_distances) / (max_distance - min_distance)) * 100
    return similarity_percentage

@app.route("/testing_image", methods=["POST"])
def testing_image():
    global image

    # Memuat model untuk setiap skenario
    model_scenario_1 = joblib.load('model_scenario_1.pkl')
    model_scenario_2 = joblib.load('model_scenario_2.pkl')
    model_scenario_3 = joblib.load('model_scenario_3.pkl')

    features = extract_features(image)

    # Menginisialisasi hasil prediksi dan persentase keputusan untuk setiap skenario
    predictions = []

    # Prediksi kelas untuk setiap skenario
    predicted_class_scenario_1 = model_scenario_1.predict([features])[0]
    predicted_class_scenario_2 = model_scenario_2.predict([features])[0]
    predicted_class_scenario_3 = model_scenario_3.predict([features])[0]

    # Menghitung persentase keputusan untuk setiap skenario
    decision_scores_scenario_1 = model_scenario_1.decision_function([features])[0]
    decision_scores_scenario_2 = model_scenario_2.decision_function([features])[0]
    decision_scores_scenario_3 = model_scenario_3.decision_function([features])[0]
    decision_percentage_scenario_1 = (decision_scores_scenario_1 - decision_scores_scenario_1.min()) / (decision_scores_scenario_1.max() - decision_scores_scenario_1.min()) * 100
    decision_percentage_scenario_2 = (decision_scores_scenario_2 - decision_scores_scenario_2.min()) / (decision_scores_scenario_2.max() - decision_scores_scenario_2.min()) * 100
    decision_percentage_scenario_3 = (decision_scores_scenario_3 - decision_scores_scenario_3.min()) / (decision_scores_scenario_3.max() - decision_scores_scenario_3.min()) * 100

    # Menambahkan hasil prediksi dan persentase keputusan ke dalam daftar predictions
    predictions.append({
        "scenario": "scenario_1",
        "result": predicted_class_scenario_1,
        "decision_percentage": get_decision_percentage(predicted_class_scenario_1, decision_percentage_scenario_1),
        "other_disease_percentages": get_other_disease_percentages(decision_percentage_scenario_1)
    })
    predictions.append({
        "scenario": "scenario_2",
        "result": predicted_class_scenario_2,
        "decision_percentage": get_decision_percentage(predicted_class_scenario_2, decision_percentage_scenario_2),
        "other_disease_percentages": get_other_disease_percentages(decision_percentage_scenario_2)
    })
    predictions.append({
        "scenario": "scenario_3",
        "result": predicted_class_scenario_3,
        "decision_percentage": get_decision_percentage(predicted_class_scenario_3, decision_percentage_scenario_3),
        "other_disease_percentages": get_other_disease_percentages(decision_percentage_scenario_3)
    })

    # Menampilkan hasil prediksi dan persentase keputusan ke dalam format JSON
    result = {
        "predictions": predictions
    }
    
    print(result)
    for prediction in predictions:
        print("Prediksi Penyakit (" + prediction["scenario"] + "):", prediction["result"])
        print("Persentase Keputusan (" + prediction["scenario"] + "):", prediction["decision_percentage"])

        print("Persentase Penyakit Lain (" + prediction["scenario"] + "):")
        for disease, percentage in prediction["other_disease_percentages"].items():
            print(disease + ":", percentage)

    return jsonify(result)


def get_decision_percentage(predicted_class, decision_percentage):
    class_names = ['Bercak Daun', 'Hawar Daun', 'Karat Daun', 'Sehat']
    class_index = class_names.index(predicted_class)
    return decision_percentage[class_index]


def get_other_disease_percentages(decision_percentage):
    disease_names = ['Bercak Daun', 'Hawar Daun', 'Karat Daun', 'Sehat']
    disease_percentages = {}
    for i, percentage in enumerate(decision_percentage):
        disease = disease_names[i]
        disease_percentages[disease] = percentage
    return disease_percentages


def get_decision_percentage(predicted_class, decision_percentage):
    class_names = ['Bercak Daun', 'Hawar Daun', 'Karat Daun', 'Sehat']
    class_index = class_names.index(predicted_class)
    return decision_percentage[class_index]



    
def extract_features_from_class(class_name):
    dataset_path = 'dataset'
    test_path = os.path.join(dataset_path, 'test', class_name.replace(" ", "-").lower())
    image_files = os.listdir(test_path)
    image = Image.open(os.path.join(test_path, image_files[0]))
    features = extract_features(image)
    return features





# Fungsi untuk membuat jendela aplikasi menggunakan webview
# def create_window():
#     webview.create_window("Deteksi Penyakit Daun Jagung", app, width=1500, height=1000, resizable=True)


# if __name__ == "__main__":
#     from threading import Thread
#     thread = Thread(target=create_window)
#     thread.start()

#     # Jalankan aplikasi Pywebview
#     webview.start()

if __name__ == "__main__":
    app.run(debug=True)