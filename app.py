import cv2
import numpy as np
from scipy.stats import moment
from skimage.feature import greycomatrix, greycoprops
from flask import Flask, request, render_template

app = Flask(__name__)

def calculate_color_moments(image):
    # Memisahkan citra menjadi saluran warna
    channels = cv2.split(image)
    
    color_moments = []
    
    for channel in channels:
        # Menghitung momen warna (rata-rata, simpangan baku, skewness) untuk setiap saluran
        moments = moment(channel.flatten(), moment=[0, 1, 2])
        color_moments.extend(moments)
    
    return color_moments

def calculate_glcm_features(image):
    # Mengubah citra ke skala keabuan
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Menghitung matriks GLCM
    glcm = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    
    # Menghitung fitur GLCM (kontras, homogenitas, energi, korelasi)
    kontras = greycoprops(glcm, 'contrast')[0, 0]
    homogenitas = greycoprops(glcm, 'homogeneity')[0, 0]
    energi = greycoprops(glcm, 'energy')[0, 0]
    korelasi = greycoprops(glcm, 'correlation')[0, 0]
    
    return kontras, homogenitas, energi, korelasi

def detect_disease(color_moments, kontras, homogenitas, korelasi, energi):
    # Logika deteksi penyakit berdasarkan nilai fitur
    penyakit = ""
    
    if kontras > 100 and homogenitas > 0.8:
        penyakit = "Penyakit A"
    elif korelasi < 0.3 and energi > 0.5:
        penyakit = "Penyakit B"
    else:
        penyakit = "Tidak Terdeteksi Penyakit"
    
    return penyakit

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Mendapatkan file gambar yang diunggah
        file = request.files['image']
        
        # Membaca citra
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Menghitung momen warna
        color_moments = calculate_color_moments(image)
        
        # Menghitung fitur GLCM
        kontras, homogenitas, energi, korelasi = calculate_glcm_features(image)
        
        # Deteksi penyakit
        penyakit = detect_disease(color_moments, kontras, homogenitas, korelasi, energi)
        
        # Merender template dengan hasil
        return render_template('result.html', kontras=kontras, homogenitas=homogenitas,
                               korelasi=korelasi, energi=energi, penyakit=penyakit)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)