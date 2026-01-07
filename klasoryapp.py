import os
import shutil

# Ana dizin yolu
root_folder = r'C:\Users\Bemol\Documents\GitHub\HearMeMioConnect\deletedrows'

# Katılımcı klasörlerini listele (sadece dizin olanları ve gizli olmayanları al)
try:
    participants = [d for d in os.listdir(root_folder) 
                   if os.path.isdir(os.path.join(root_folder, d)) and not d.startswith('.')]

    for p_name in participants:
        p_folder = os.path.join(root_folder, p_name)
        
        # Klasördeki tüm .npy dosyalarını bul
        files = [f for f in os.listdir(p_folder) if f.endswith('.npy')]

        for fname in files:
            # Gesture ismini dosya adından çıkar (split işlemi)
            # Örn: "yumruk_01.npy" -> "yumruk"
            parts = fname.split('_')
            gesture_name = parts[0]

            # Gesture klasörü yolu
            gesture_folder = os.path.join(p_folder, gesture_name)

            # Klasör yoksa oluştur
            if not os.path.exists(gesture_folder):
                os.makedirs(gesture_folder)

            # Dosyayı taşı
            source_path = os.path.join(p_folder, fname)
            destination_path = os.path.join(gesture_folder, fname)
            
            shutil.move(source_path, destination_path)

    print("Dosyalar gesture klasörlerine taşındı.")

except FileNotFoundError:
    print(f"Hata: {root_folder} dizini bulunamadı.")