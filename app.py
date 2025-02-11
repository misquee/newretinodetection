from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
import numpy as np
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from reportlab.pdfgen import canvas
import uuid
from flask_cors import CORS  # Tambahkan untuk penanganan CORS

app = Flask(__name__)
CORS(app)  # Izinkan semua domain untuk menghindari masalah CORS

# Konfigurasi folder upload
UPLOAD_FOLDER = "upload"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model deep learning
MODEL_PATH = "model_percobaanx.h5"  # Sesuaikan dengan lokasi model Anda
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise ValueError(f"Error loading model: {e}")

# Label prediksi
LABELS = ["RB", "Normal"]  # RB = Retinoblastoma

# Fungsi untuk memeriksa ekstensi file yang diizinkan
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def predict():
    if "image" in request.files:
        # Jika file gambar dikirim sebagai file biasa
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File format not supported"}), 400

        # Generate nama file unik
        unique_id = str(uuid.uuid4())[:8]  # Ambil 8 karakter pertama dari UUID
        filename = secure_filename(f"{unique_id}_{file.filename}")
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        print(f"File saved: {filepath}")  # Log lokasi file
    elif "image_base64" in request.form:
        # Jika gambar dikirim dalam bentuk base64
        image_data = request.form["image_base64"]
        try:
            if "," in image_data:
                _, encoded = image_data.split(",", 1)  # Hapus prefix data:image/jpeg;base64,
            else:
                encoded = image_data

            binary_data = base64.b64decode(encoded)
            unique_id = str(uuid.uuid4())[:8]  # Ambil 8 karakter pertama dari UUID
            filename = f"{unique_id}_camera_upload.jpg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            with open(filepath, "wb") as f:
                f.write(binary_data)
            print(f"Base64 image saved: {filepath}")  # Log lokasi file
        except Exception as e:
            return jsonify({"error": f"Error decoding base64 image: {e}"}), 500
    else:
        return jsonify({"error": "No image provided"}), 40

    # Preprocessing gambar
    try:
        img = load_img(filepath, target_size=(224, 224))  # Sesuaikan ukuran input model
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 500

    # Prediksi
    try:
        predictions = model.predict(img_array)
        confidence = predictions[0]
        result_index = np.argmax(confidence)
        result_label = LABELS[result_index]
        result_confidence = confidence[result_index]
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {e}"}), 500

    # Simpan hasil ke PDF
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{filename.rsplit('.', 1)[0]}.pdf")
    try:
        generate_pdf(result_label, result_confidence, filepath, pdf_path)
    except Exception as e:
        return jsonify({"error": f"Error generating PDF: {e}"}), 500

    # Kirim hasil prediksi beserta URL gambar
    return jsonify({
        "result": result_label,
        "confidence": float(result_confidence),
        "image_url": f"/upload/{filename}",
        "pdf_url": f"/upload/{os.path.basename(pdf_path)}"
    })

def generate_pdf(result, confidence, image_path, output_path):
    try:
        c = canvas.Canvas(output_path)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, 750, "Hasil Deteksi Retinoblastoma")
        c.setFont("Helvetica", 12)
        c.drawString(100, 720, f"Status: {result}")
        c.drawString(100, 700, f"Kepercayaan: {(confidence * 100):.2f}%")
        c.drawImage(image_path, 100, 450, width=300, height=300)
        c.save()
    except Exception as e:
        raise ValueError(f"Error generating PDF: {e}")

@app.route("/upload/<filename>")
def uploaded_file(filename):
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)