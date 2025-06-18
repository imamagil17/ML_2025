import csv
import numpy as np
import re
import random

# Label ke angka
label_map = {'DN': 0, 'DD': 1, 'B': 2, 'D': 3}
reverse_map = {v: k for k, v in label_map.items()}

# Label ke daerah
label_region = {
    'DN': 'Sulawesi Tengah',
    'DD': 'Sulawesi Selatan',
    'B': 'DKI Jakarta',
    'D': 'Bandung'
}

# Plat yang diizinkan
allowed_prefixes = set(label_map.keys())

# Ekstrak dua huruf awal sebagai fitur
def extract_features(plat):
    plat = plat.strip().upper()
    prefix = re.match(r'^[A-Z]{1,2}', plat)
    if not prefix:
        return [-1, -1]
    prefix = prefix.group().rjust(2)
    return [ord(prefix[0]) - ord('A'), ord(prefix[1]) - ord('A')]

# Cek apakah prefix valid
def is_valid_prefix(plat):
    plat = plat.strip().upper()
    match = re.match(r'^[A-Z]{1,2}', plat)
    if not match:
        return False
    prefix = match.group()
    return prefix in allowed_prefixes

# Baca dataset dari CSV
def load_data_from_csv(filename):
    X = []
    y = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['label'] not in label_map:
                continue
            feat = extract_features(row['plat'])
            if -1 in feat:
                continue
            X.append(feat)
            y.append(label_map[row['label']])
    return np.array(X), np.array(y)

# Bagi data
def split_data(X, y, test_ratio=0.2):
    total = len(X)
    idx = list(range(total))
    random.shuffle(idx)
    split = int(total * (1 - test_ratio))
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]

# KNN sederhana
def predict_knn(X_train, y_train, x_input, k=3):
    distances = np.linalg.norm(X_train - x_input, axis=1)
    k_idx = distances.argsort()[:k]
    k_labels = y_train[k_idx]
    return np.bincount(k_labels).argmax()

# Evaluasi akurasi
def evaluate(X_train, y_train, X_test, y_test):
    correct = 0
    for xi, yi in zip(X_test, y_test):
        pred = predict_knn(X_train, y_train, xi)
        if pred == yi:
            correct += 1
    acc = correct / len(y_test)
    print(f"\nAkurasi model terhadap data uji: {acc * 100:.2f}%")

# Input manual
def manual_test(X_train, y_train):
    while True:
        plat = input("\nMasukkan plat nomor (atau ketik 'exit'): ").strip().upper()
        if plat == 'EXIT':
            break
        if not is_valid_prefix(plat):
            print("Prefix plat tidak dikenali. Hanya menerima DN, DD, D, dan B.")
            continue
        feat = extract_features(plat)
        if -1 in feat:
            print("Format plat nomor tidak dikenali.")
            continue
        pred = predict_knn(X_train, y_train, np.array(feat))
        kode = reverse_map[pred]
        daerah = label_region.get(kode, 'Daerah tidak diketahui')
        print(f"Plat Nomor: {plat} → Diprediksi berasal dari: {kode} ({daerah})")

# Main
def main():
    print("Membaca data dari file 'dataset_manual.csv'...")
    X, y = load_data_from_csv('dataset_manual.csv')

    if len(X) == 0:
        print("Data tidak ditemukan atau format salah.")
        return

    X_train, y_train, X_test, y_test = split_data(X, y)

    print("Melatih dan menguji model...")
    evaluate(X_train, y_train, X_test, y_test)

    print("\nUji plat nomor manual:")
    manual_test(X_train, y_train)

if __name__ == '__main__':
    main()
