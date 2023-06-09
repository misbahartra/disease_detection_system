from flask import Flask, render_template, request, url_for, send_from_directory
from PIL import Image
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import os
import time

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
    global image, counter
    image_file = request.files["image"]
    image = Image.open(image_file)
    timestamp = int(time.time())
    image_name = f"image_{timestamp}.jpg"
    image_path = "uploads/" + image_name
    image.save(image_path)  # Simpan gambar di folder uploads
    image_url = "/uploads/" + image_name  # URL gambar yang akan ditampilkan di halaman result
    counter += 1
    return image_url


@app.route("/grayscale", methods=["POST"])
def convert_to_grayscale_route():
    global image, counter, gray_image
    gray_image = image.convert("L")
    timestamp = int(time.time())
    filename = f"gray_image_{timestamp}.jpg"
    gray_image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return filename

@app.route("/glcm", methods=["POST"])
def calculate_glcm():
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

    return result

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
    return result

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)