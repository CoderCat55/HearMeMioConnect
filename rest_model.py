from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

class RestDetector:
    def __init__(self, window_size=20):
        self.model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        self.scaler = StandardScaler()
        self.window_size = window_size

    @staticmethod
    def extract_features(time_series_data):
        features = []
        for channel in range(time_series_data.shape[1]):
            channel_data = time_series_data[:, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.max(channel_data) - np.min(channel_data),
            ])
        return np.array(features)

    def train(self, base_path='deletedrows'):
        """
        Asimetrik klasör yapısını tarar:
        1. restpX -> Direkt .npy dosyaları (REST)
        2. pX -> Alt klasörler (bek, yumruk vb.) -> .npy dosyaları (ACTIVE)
        """
        rest_samples = []
        not_rest_samples = []
        
        if not os.path.exists(base_path):
            print(f"HATA: {base_path} dizini bulunamadı!")
            return False

        print(f"Veriler taranıyor: {base_path}...")
        
        for folder_name in os.listdir(base_path):
            full_path = os.path.join(base_path, folder_name)
            if not os.path.isdir(full_path): continue
            
            # --- DURUM 1: restp1, restp2 gibi REST klasörleri ---
            if folder_name.startswith("rest"):
                print(f"  [REST] Klasörü taranıyor: {folder_name}")
                for file in os.listdir(full_path):
                    if file.endswith(".npy"):
                        data = np.load(os.path.join(full_path, file))
                        rest_samples.append(data)

            # --- DURUM 2: p1, p2, p3 gibi JEST (Active) klasörleri ---
            elif folder_name.startswith("p"):
                print(f"  [ACTIVE] Klasörü taranıyor: {folder_name}")
                # p3'ün içindeki 'bek', 'yumruk' gibi alt klasörlere gir
                for sub_folder in os.listdir(full_path):
                    sub_path = os.path.join(full_path, sub_folder)
                    if os.path.isdir(sub_path):
                        for file in os.listdir(sub_path):
                            if file.endswith(".npy"):
                                data = np.load(os.path.join(sub_path, file))
                                not_rest_samples.append(data)

        if len(rest_samples) == 0 or len(not_rest_samples) == 0:
            print(f"HATA: Veri bulunamadı! Rest dosyası: {len(rest_samples)}, Aktif dosyası: {len(not_rest_samples)}")
            return False

        # Özellik çıkarma (Sliding Window)
        X, y = [], []
        stride = self.window_size // 2
        
        print(f"Özellikler çıkarılıyor... (Toplam {len(rest_samples) + len(not_rest_samples)} dosya)")
        
        for samples, label in [(rest_samples, 0), (not_rest_samples, 1)]:
            for time_series in samples:
                num_windows = (len(time_series) - self.window_size) // stride + 1
                if num_windows <= 0: continue
                
                for i in range(num_windows):
                    window = time_series[i*stride : i*stride + self.window_size]
                    X.append(self.extract_features(window))
                    y.append(label)

        X = np.array(X)
        y = np.array(y)
        
        # Normalizasyon ve Eğitim
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        print(f"✓ RestDetector Başarıyla Eğitildi!")
        print(f"  - Toplam Pencere (Window): {len(X)}")
        print(f"  - Sınıf Dağılımı: Rest={np.sum(y==0)}, Active={np.sum(y==1)}")
        return True
    def predict(self, window_data):
        features = self.extract_features(window_data).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        return prediction[0] == 0

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'window_size': self.window_size}, f)

    def load_model(self, filepath):
        if not os.path.exists(filepath): return False
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model, self.scaler, self.window_size = data['model'], data['scaler'], data['window_size']
        return True