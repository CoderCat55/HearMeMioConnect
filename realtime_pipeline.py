import time
import numpy as np
import pickle
import os
from model1 import RestDetector
from model2 import GestureClassifierSVM

# ==============================================================================
# AYARLAR
# ==============================================================================
WINDOW_SIZE_DETECTION = 20   # Model 1 için pencere boyutu (Rest tespiti)
WINDOW_SIZE_CLASSIFY = 200   # Model 2 için toplanacak veri boyutu (Sınıflandırma)
MODEL2_PATH = "model2.pkl"
# ==============================================================================

def load_model2(path):
    if not os.path.exists(path):
        print(f"HATA: {path} bulunamadı. Önce model2.py'yi çalıştırıp eğitin.")
        return None
    
    classifier = GestureClassifierSVM()
    with open(path, 'rb') as f:
        data = pickle.load(f)
        classifier.svm = data['model']
        classifier.scaler = data['scaler']
        classifier.is_trained = True
    return classifier

def get_data_chunk(buffer, current_index, num_samples):
    """
    Simülasyon: Paylaşılan bellekten (buffer) son 'num_samples' kadar veri okur.
    Gerçek main.py entegrasyonunda burası shared_memory okuması yapacak.
    """
    # NOT: Bu fonksiyon main.py'deki 'get_recent_data_from_shared_mem' mantığıyla çalışmalıdır.
    # Şimdilik buffer'ın sonundan veri çekiyormuş gibi davranıyoruz.
    
    # Eğer buffer numpy array ise:
    if isinstance(buffer, np.ndarray):
        # Buffer boyutu yetersizse
        if len(buffer) < num_samples:
            return buffer
        return buffer[-num_samples:]
    
    # Mock data (Test için rastgele veri üretir)
    return np.random.rand(num_samples, 34).astype(np.float32)

def run_pipeline(stream_buffer, stream_index, is_running):
    """
    Kullanıcının istediği Pipeline Mantığı:
    1. 20 sample oku -> Model 1 (Rest mi?)
    2. Eğer Rest ise -> Başa dön (continue)
    3. Eğer Aktif ise -> 200 sample topla -> Model 2 (Sınıflandır)
    """
    print("--- Pipeline Başlatılıyor ---")
    
    # 1. Model 1'i Hazırla (RestDetector)
    # Not: Model 1'in eşik değeri (threshold) daha önce hesaplanmış olmalı.
    # Burada varsayılan veya kaydedilmiş bir değer kullanılabilir.
    # Gerçek senaryoda: detector.fit(rest_data) yapılmış olmalı.
    detector = RestDetector(window_size=WINDOW_SIZE_DETECTION)
    # Örnek: Threshold manuel set edilebilir veya bir dosyadan yüklenebilir.
    # detector.threshold = 0.05 
    
    # 2. Model 2'yi Yükle (SVM)
    classifier = load_model2(MODEL2_PATH)
    if classifier is None:
        return

    print("Modeller yüklendi. Döngü başlıyor...")

    while is_running.value: # Kullanıcı durdurana kadar
        
        # --- ADIM 1: 20 Sample Oku ---
        # Gerçek sistemde burada buffer'ın dolmasını beklemek gerekebilir.
        detection_window = get_data_chunk(stream_buffer, stream_index, WINDOW_SIZE_DETECTION)
        
        # Veri yetersizse bekle
        if len(detection_window) < WINDOW_SIZE_DETECTION:
            time.sleep(0.01)
            continue

        # --- ADIM 2: Model 1 Kontrolü (Rest mi?) ---
        # detector.predict -> True dönerse REST, False dönerse ACTIVE
        is_rest = detector.predict(detection_window)
        
        if is_rest:
            # Rest pozisyonu, işlem yapma, başa dön
            # print(".", end="", flush=True) # Debug için nokta koyabiliriz
            time.sleep(0.05) # İşlemciyi yormamak için kısa bekleme
            continue
        
        # --- ADIM 3: Hareket Algılandı! Veri Topla ---
        print("\n! Hareket Algılandı -> Veri toplanıyor...")
        
        # 200 sample (veya hareket bitene kadar) bekle
        # Gerçek zamanlı sistemde bu bekleme süresi sample rate'e göre ayarlanmalı
        # Örn: 200 sample @ 100Hz = 2 saniye
        time.sleep(2.0) 
        
        classification_window = get_data_chunk(stream_buffer, stream_index, WINDOW_SIZE_CLASSIFY)
        
        # --- ADIM 4: Model 2 Sınıflandırma ---
        prediction = classifier.predict(classification_window)
        
        print(f">>> TAHMİN: {prediction}")
        
        # Tahmin sonrası tekrar aynı hareketi yakalamamak için kısa bir bekleme (Debounce)
        time.sleep(1.0)