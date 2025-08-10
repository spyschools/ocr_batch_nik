#!/usr/bin/env python3
import cv2
import pytesseract
import numpy as np
import re
import csv
from pathlib import Path

# ==== Konfigurasi ====
folder_path = "images"         # Folder tempat gambar
output_text_file = "hasil_ocr.txt"  # Output teks lengkap
output_csv_file = "nik_list.csv"    # Output CSV berisi NIK saja

# ==== Fungsi Auto-Rotate berdasarkan deteksi teks ====
def auto_rotate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# ==== Fungsi Sharpen Gambar ====
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# ==== Pencarian gambar ====
folder = Path(folder_path)
if not folder.exists():
    raise FileNotFoundError(f"Folder '{folder_path}' tidak ditemukan!")

image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
if not image_files:
    raise FileNotFoundError(f"Tidak ada gambar .jpg/.png/.jpeg di folder '{folder_path}'")

print(f"🔍 Ditemukan {len(image_files)} gambar di '{folder_path}'")
print("📖 Memulai proses OCR...\n")

# ==== Variabel untuk hasil ====
all_results = []
nik_results = []

# ==== Proses setiap gambar ====
for img_path in image_files:
    print(f"📂 Memproses: {img_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Gagal membaca {img_path.name}, lewati...")
        continue

    # Preprocessing: auto-rotate + sharpen
    img = auto_rotate(img)
    img = sharpen_image(img)

    # Grayscale & threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)

    # OCR
    result = pytesseract.image_to_string(threshed, lang="ind")

    # Normalisasi teks
    normalized_lines = []
    for word in result.split("\n"):
        if "”—" in word:
            word = word.replace("”—", ":")
        if "NIK" in word:
            word = word.replace("D", "0").replace("?", "7")
        if word.strip():
            normalized_lines.append(word)

    # Simpan teks lengkap
    all_results.append(f"===== {img_path.name} =====\n" + "\n".join(normalized_lines) + "\n")

    # ==== Deteksi NIK (16 digit angka) ====
    nik_pattern = r"\b\d{16}\b"
    found_nik = re.findall(nik_pattern, "\n".join(normalized_lines))
    if found_nik:
        for nik in found_nik:
            nik_results.append({"file": img_path.name, "nik": nik})

# ==== Simpan hasil OCR lengkap ====
with open(output_text_file, "w", encoding="utf-8") as f:
    f.write("\n".join(all_results))

# ==== Simpan NIK ke CSV ====
if nik_results:
    with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["file", "nik"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in nik_results:
            writer.writerow(row)

print(f"\n✅ OCR selesai!")
print(f"📄 Hasil teks lengkap: '{output_text_file}'")
if nik_results:
    print(f"📊 NIK terdeteksi ({len(nik_results)}): disimpan di '{output_csv_file}'")
else:
    print("⚠️ Tidak ada NIK yang terdeteksi.")
