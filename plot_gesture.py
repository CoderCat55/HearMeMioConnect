"""
Gesture Plotter
---------------
Bu script, işlenmiş (.npy) dosyalarını görselleştirmek için kullanılır.
EMG sinyallerini ve hesaplanan enerji seviyesini grafiğe döker.
Böylece rest (dinlenme) kısımlarının kesilip kesilmediğini gözle kontrol edebilirsiniz.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import sys

def load_data(file_path):
    try:
        raw_content = np.load(file_path, allow_pickle=True)
        # DataHandler formatı veya düz array kontrolü
        if isinstance(raw_content, np.ndarray) and raw_content.ndim == 2 and np.issubdtype(raw_content.dtype, np.number):
            return raw_content.astype(np.float32)
        try:
            if len(raw_content) > 0 and len(raw_content[0]) == 2 and isinstance(raw_content[0][1], (np.ndarray, list)):
                data_list = [x[1] for x in raw_content]
                return np.vstack(data_list).astype(np.float32)
            return np.vstack(raw_content).astype(np.float32)
        except:
            return np.array(raw_content).astype(np.float32)
    except Exception as e:
        print(f"Dosya yükleme hatası: {e}")
        return None

def plot_file(file_path):
    if not os.path.exists(file_path):
        print(f"HATA: Dosya bulunamadı -> {file_path}")
        print("Lütfen kodun altındaki 'TARGET_FILE' değişkenine geçerli bir dosya yolu girin.")
        return

    data = load_data(file_path)
    if data is None:
        return

    # EMG Kolonlarını seç (Myo1: 0-7, Myo2: 17-24)
    n_cols = data.shape[1]
    emg_cols = []
    if n_cols >= 8: emg_cols.extend(range(0, 8))
    if n_cols >= 25: emg_cols.extend(range(17, 25))
    
    if not emg_cols:
        print("Uyarı: EMG kolonları tespit edilemedi, tüm veri çiziliyor.")
        emg_data = data
    else:
        emg_data = data[:, emg_cols]

    # Enerji Hesapla (Kesme işleminde kullanılan mantıkla aynı)
    abs_data = np.abs(emg_data)
    mean_energy = np.mean(abs_data, axis=1)
    window_size = 20
    window = np.ones(window_size) / window_size
    smoothed_energy = np.convolve(mean_energy, window, mode='same')

    # Grafik Çiz
    plt.figure(figsize=(12, 8))

    # 1. Alt Grafik: Tüm EMG Kanalları
    plt.subplot(2, 1, 1)
    plt.plot(emg_data, alpha=0.6, linewidth=0.8)
    plt.title(f"Ham EMG Sinyalleri - {os.path.basename(file_path)}")
    plt.ylabel("Genlik")
    plt.grid(True, alpha=0.3)

    # 2. Alt Grafik: Enerji Seviyesi
    plt.subplot(2, 1, 2)
    plt.plot(smoothed_energy, color='red', linewidth=2, label='Ortalama Enerji')
    plt.title("Sinyal Enerjisi (Hareket Aktivitesi)")
    plt.xlabel("Sample Sayısı")
    plt.ylabel("Enerji")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
    
    target_file = None

    # 1. Komut satırından dosya yolu verildiyse onu kullan
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    
    # 2. If not provided, find a random .npy file from the processed_data folder
    if target_file is None or not os.path.exists(target_file):
        npy_files = []
        for root, dirs, files in os.walk(PROCESSED_DIR):
            for file in files:
                if file.endswith(".npy"):
                    npy_files.append(os.path.join(root, file))
        
        if npy_files:
            target_file = random.choice(npy_files)
        else:
            print("HATA: İşlenmiş .npy dosyası bulunamadı. 'processed_data' klasörünü kontrol edin.")
            sys.exit(1)
        
    if target_file and os.path.exists(target_file):
        print(f"Görselleştiriliyor: {target_file}")

        # Açıklama satırını yazdır
        print("NOT: Başarılı bir segmentasyonda, grafiğin başında ve sonunda uzun düz (rest) çizgiler olmamalıdır.")
        plot_file(target_file)
    else:
        print("HATA: Gösterilecek dosya bulunamadı. Lütfen processed_data klasörünü kontrol edin.")