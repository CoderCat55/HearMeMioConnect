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
    # BURAYI DÜZENLEYİN: Kontrol etmek istediğiniz dosyanın tam yolunu buraya yapıştırın.
    TARGET_FILE = r"C:\Users\yagmu\OneDrive\Belgeler\GitHub\HearMeMioConnect\processed_data\p1\yrd_1765448698_cropped.npy"
    
    # Eğer dosyayı terminalden argüman olarak verirseniz onu kullanır
    if len(sys.argv) > 1:
        TARGET_FILE = sys.argv[1]
        
    print(f"Görselleştiriliyor: {TARGET_FILE}")
    plot_file(TARGET_FILE)