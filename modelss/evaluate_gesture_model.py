import numpy as np
import joblib
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from datetime import datetime  
import sys 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Configuration ---
DATA_DIRECTORY = "deletedrows"  # Root folder containing gesture subfolders
WINDOW_SIZE = 100
WINDOW_STEP = 50

class GestureClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.gesture_map = {}
        self.label_to_gesture = None

    def organize_files(self, base_path):
        """
        Matlab kodunun yaptığı işi yapar: 
        Dosya adındaki ilk kelimeyi (örn: 'yumruk_1.npy') klasör adı yapar 
        ve dosyayı o klasöre taşır.
        """
        print("Klasörler düzenleniyor...")
        
        # Katılımcı klasörlerini bul (p1, p2 vb.)
        participants = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
        for p_name in participants:
            p_folder = os.path.join(base_path, p_name)
            
            # Klasör içindeki .npy dosyalarını bul
            files = [f for f in os.listdir(p_folder) if f.lower().endswith('.npy') and os.path.isfile(os.path.join(p_folder, f))]
            
            if not files:
                continue
                
            for fname in files:
                # Matlab'daki: parts = split(fname, '_'); gestureName = parts{1};
                gesture_name = fname.split('_')[0]
                
                # Hedef klasör yolu: deletedrows/p1/yumruk
                gesture_folder = os.path.join(p_folder, gesture_name)
                
                # klasör yoksa oluştur (mkdir)
                os.makedirs(gesture_folder, exist_ok=True)
                
                # dosyayı taşı (movefile)
                src_path = os.path.join(p_folder, fname)
                dst_path = os.path.join(gesture_folder, fname)
                
                try:
                    shutil.move(src_path, dst_path)
                except Exception as e:
                    print(f"Hata: {fname} taşınamadı. {e}")

        print("Düzenleme tamamlandı. Dosyalar ilgili gesture klasörlerine taşındı.")

    @staticmethod
    def extract_features(data, window_size=WINDOW_SIZE, window_step=WINDOW_STEP):
        """
        Extracts time-domain features from sEMG data using a sliding window.
        This MUST match the feature extraction in trainmodel.py.
        """
        n_samples, n_channels = data.shape
        features = []

        for start in range(0, n_samples - window_size + 1, window_step):
            end = start + window_size
            window = data[start:end, :]

            window_features = []
            for i in range(n_channels):
                channel_data = window[:, i]

                # 1. Mean Absolute Value (MAV)
                mav = np.mean(np.abs(channel_data))
                # 2. Waveform Length (WL)
                wl = np.sum(np.abs(np.diff(channel_data)))
                # 3. Zero Crossings (ZC)
                zc = np.sum(np.diff(np.sign(channel_data)) != 0)
                # 4. Slope Sign Changes (SSC)
                ssc = np.sum(np.diff(np.sign(np.diff(channel_data))) != 0)

                window_features.extend([mav, wl, zc, ssc])
            features.append(window_features)
        return np.array(features)

    def _load_nested_data(self, base_path):
        """
        Tüm alt klasörleri (p1, p2 vb.) gezer ve içindeki gesture klasörlerini bulur.
        Klasör yapısı: base_path / katılımcı_klasörü / gesture_klasörü / dosyalar.npy
        """
        all_features, all_labels = [], []
        
        print(f"Scanning directory: {base_path}")
        
        # os.walk kullanarak tüm alt dizinleri derinlemesine tarıyoruz
        for root, dirs, files in os.walk(base_path):
            # Sadece içinde .npy dosyası olan klasörlerle ilgileniyoruz
            npy_files = [f for f in files if f.lower().endswith('.npy')]
            
            if not npy_files:
                continue

            # Klasörün tam yolunun son parçası gesture adıdır (örn: "yumruk")
            gesture_name = os.path.basename(root)
            
            if gesture_name not in self.gesture_map:
                self.gesture_map[gesture_name] = len(self.gesture_map)
            
            label = self.gesture_map[gesture_name]
            print(f"  - Bulundu: '{gesture_name}' klasörü -> {len(npy_files)} dosya")
            
            for file_name in npy_files:
                data = np.load(os.path.join(root, file_name))
                if data.size > 0:
                    feats = self.extract_features(data)
                    if feats.shape[0] > 0:
                        all_features.append(feats)
                        all_labels.append(np.full(feats.shape[0], label))

        if not all_features:
            print("Hata: Hiçbir .npy dosyası veya gesture klasörü bulunamadı!")
            return None, None, None

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        return X, y, self.gesture_map

    def run_full_training(self):
        # 0. Dosya adını ve tarih formatını hazırla (offlinemodelcomparison.ddmmyyyy.txt)
        date_str = datetime.now().strftime("%d%m%Y")
        report_filename = f"offlinemodelcomparison.{date_str}.txt"

        # 0. Önce dosyaları Matlab mantığıyla düzenle (Dosya adından klasör oluşturma)
        self.organize_files(DATA_DIRECTORY)

        # 1. Load Data
        X, y, gesture_map = self._load_nested_data(DATA_DIRECTORY)
        if X is None: 
            print("Veri yüklenemedi!")
            return

        # 2. Split and Scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # 3. Define Models (6 modelden 5'i aktif - Gradient Boosting eklenebilir)
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "K-Nearest Neighbors": KNeighborsClassifier(n_jobs=-1),
            "Support Vector Machine": SVC(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1)
        }
        
        results = {}

        # 4. Train, Compare & Write to Text File
        with open(report_filename, "w", encoding="utf-8") as f:
            f.write(f"OFFLINE MODEL KARŞILAŞTIRMA RAPORU - {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write("="*90 + "\n")
            f.write(f"{'Model Adı':<25} | {'Doğruluk':<10} | {'Hiperparametreler'}\n")
            f.write("-" * 90 + "\n")

            print("\n" + "="*40)
            print("MODEL PERFORMANCE COMPARISON")
            print("="*40)

            for name, model in models.items():
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))
                results[name] = acc
                
                # Model parametrelerini metne dönüştür
                params_str = str(model)
                
                # Hem ekrana hem dosyaya yazdır
                output_line = f"{name:25}: {acc*100:.2f}%"
                print(output_line)
                f.write(f"{name:<25} | {acc*100:>8.2f}% | {params_str}\n")

            # 5. Best Model Detailed Report
            best_name = max(results, key=results.get)
            print(f"\n>>> Best Model: {best_name}")
            
            self.model = models[best_name]
            y_pred = self.model.predict(X_test)
            
            self.label_to_gesture = {v: k for k, v in gesture_map.items()}
            target_names = [self.label_to_gesture[i] for i in sorted(self.label_to_gesture.keys())]

            detail_report = classification_report(y_test, y_pred, target_names=target_names)
            
            # Detaylı raporu dosyaya ekle
            f.write("\n" + "="*90 + "\n")
            f.write(f">>> EN İYİ MODEL: {best_name}\n")
            f.write("\nDetailed Classification Report:\n")
            f.write(detail_report)

            print("\nDetailed Classification Report:")
            print(detail_report)

        # 6. Confusion Matrix Visualization
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=target_names, yticklabels=target_names)
        plt.title(f'Final Confusion Matrix: {best_name}')
        plt.ylabel('Actual Gesture')
        plt.xlabel('Predicted Gesture')
        plt.tight_layout()
        plt.savefig('best_model_results.png')
        
        print(f"\nSonuçlar '{report_filename}' ve 'best_model_results.png' olarak kaydedildi.")
        plt.show()

if __name__ == "__main__":
    clf = GestureClassifier()
    clf.run_full_training()