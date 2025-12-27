import os
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class GestureClassifierSVM:
    """
    Model 2: SVM tabanlı hareket sınıflandırıcı.
    Segmente edilmiş verilerden özellik çıkarır ve sınıflandırma yapar.
    """
    def __init__(self):
        # SVM Modeli (RBF kernel genellikle EMG için iyidir, ancak Linear da denenebilir)
        # probability=True, tahminlerin olasılık değerlerini görmek için
        self.svm = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_features(self, raw_data):
        """
        Ham veriden istatistiksel özellikler çıkarır.
        Girdi: (N_sample, 34)
        Çıktı: (N_features, ) -> 34 kanal * 5 özellik = 170 özellik
        """
        # Veri boşsa veya hatalıysa sıfır vektör döndür
        if raw_data is None or raw_data.size == 0:
            return np.zeros(34 * 5)

        # 1. Ortalama (Mean)
        mean = np.mean(raw_data, axis=0)
        # 2. Standart Sapma (Std Dev)
        std = np.std(raw_data, axis=0)
        # 3. Root Mean Square (RMS)
        rms = np.sqrt(np.mean(raw_data**2, axis=0))
        # 4. Minimum Değer
        min_val = np.min(raw_data, axis=0)
        # 5. Maksimum Değer
        max_val = np.max(raw_data, axis=0)

        # Tüm özellikleri tek bir vektörde birleştir
        features = np.concatenate([mean, std, rms, min_val, max_val])
        return features

    def load_data_and_train(self, processed_dir):
        """
        processed_data klasörünü tarar, verileri yükler ve modeli eğitir.
        """
        print(f"Veriler yükleniyor: {processed_dir}")
        X = []
        y = []

        # Klasörleri gez
        for root, dirs, files in os.walk(processed_dir):
            for file in files:
                if file.endswith(".npy"):
                    file_path = os.path.join(root, file)
                    
                    # Dosya isminden etiketi al (örn: yumruk_12345_seg0.npy -> yumruk)
                    label = file.split('_')[0]
                    
                    try:
                        data = np.load(file_path, allow_pickle=True)
                        if data is None or data.size == 0:
                            continue
                            
                        # Özellik çıkarımı yap
                        features = self.extract_features(data)
                        
                        X.append(features)
                        y.append(label)
                    except Exception as e:
                        print(f"Hata ({file}): {e}")

        if not X:
            print("HATA: Eğitim için veri bulunamadı!")
            return False

        X = np.array(X)
        y = np.array(y)

        print(f"Toplam Örnek Sayısı: {len(X)}")
        print(f"Sınıflar: {np.unique(y)}")

        # Veriyi Eğitim ve Test olarak ayır (%80 Eğitim, %20 Test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Veriyi ölçekle (StandardScaler)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Model eğitiliyor (SVM)...")
        self.svm.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Test başarısını ölç
        y_pred = self.svm.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nModel Doğruluğu: %{acc*100:.2f}")
        print(classification_report(y_test, y_pred))

        return True

    def predict(self, raw_data):
        """
        Yeni gelen ham veri için tahmin yapar.
        """
        if not self.is_trained:
            return "Model Eğitilmedi"
            
        features = self.extract_features(raw_data).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.svm.predict(features_scaled)
        return prediction[0]

    def save_model(self, path="model2.pkl"):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.svm, 'scaler': self.scaler}, f)
        print(f"Model kaydedildi: {path}")

if __name__ == "__main__":
    # Bu dosya doğrudan çalıştırılırsa eğitimi başlat
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    
    classifier = GestureClassifierSVM()
    if classifier.load_data_and_train(PROCESSED_DIR):
        classifier.save_model(os.path.join(BASE_DIR, "model2.pkl"))