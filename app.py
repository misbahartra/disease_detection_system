from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import os

app = Flask(__name__)
global image

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
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global image
    image_file = request.files["image"]
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)
    image = Image.open(image_path)
    return "Gambar berhasil diunggah!"


@app.route("/grayscale", methods=["POST"])
def grayscale():
    global image
    gray_image = image.convert("L")
    image = gray_image
    return "Gambar berhasil diubah menjadi grayscale!"

@app.route("/glcm", methods=["POST"])
def calculate_glcm():
    global image
    gray_image = np.array(image)
    glcm = greycomatrix(gray_image, [1], [0], symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0][0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0][0]
    energy = greycoprops(glcm, 'energy')[0][0]
    correlation = greycoprops(glcm, 'correlation')[0][0]
    result = f"Nilai Kontras: {contrast}<br>"
    result += f"Homogenitas: {homogeneity}<br>"
    result += f"Energi: {energy}<br>"
    result += f"Korelasi: {correlation}<br>"
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

if __name__ == "__main__":
    app.run(debug=True)