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

# Prefix yang diizinkan
allowed_prefixes = set(label_map.keys())

# Ekstrak dua huruf awal sebagai fitur numerik
def extract_features(plat):
    plat = plat.strip().upper()
    prefix = re.match(r'^[A-Z]{1,2}', plat)
    if not prefix:
        return [-1, -1]
    prefix = prefix.group().rjust(2)
    base = [ord(prefix[0]) - ord('A'), ord(prefix[1]) - ord('A')]

    # Tambahkan noise kecil agar ada variasi antar data
    noise = [random.choice([-1, 0, 1]), random.choice([-1, 0, 1])]
    return [base[0] + noise[0], base[1] + noise[1]]

# Validasi prefix input manual
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

# Split data lebih adil (per kelas)
def split_data(X, y, test_ratio=0.2):
    X_train, X_test, y_train, y_test = [], [], [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        random.shuffle(idx)
        split = int(len(idx) * (1 - test_ratio))
        train_idx, test_idx = idx[:split], idx[split:]
        for i in train_idx:
            X_train.append(X[i])
            y_train.append(y[i])
        for i in test_idx:
            X_test.append(X[i])
            y_test.append(y[i])
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# KNN manual
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

# Input manual oleh user
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
