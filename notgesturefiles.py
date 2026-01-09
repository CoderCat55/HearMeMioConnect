import os
import shutil

def reorganize_files():
    # Kullanıcının belirttiği ana dizin
    base_path = r"C:\Users\Bemol\Documents\GitHub\HearMeMioConnect\deletedrows"
    if not os.path.exists(base_path):
        print(f"Hata: Klasör bulunamadı -> {base_path}")
        return

    print(f"İşlem başlıyor: {base_path}")

    # deletedrows içindeki her bir participant klasörünü (p1, p2...) gez
    for participant in os.listdir(base_path):
        participant_path = os.path.join(base_path, participant)

        # Sadece klasörleri işle
        if os.path.isdir(participant_path):
            print(f"\nParticipant kontrol ediliyor: {participant}")
            
            # Participant klasörü içindeki alt klasörleri (gesture klasörleri) gez
            for gesture_folder in os.listdir(participant_path):
                gesture_path = os.path.join(participant_path, gesture_folder)

                # Eğer bu bir klasörse (örn: yrd, merhaba) içine gir
                if os.path.isdir(gesture_path):
                    # Klasördeki tüm dosyaları kontrol et
                    for filename in os.listdir(gesture_path):
                        if filename.endswith(".npy"):
                            src_file = os.path.join(gesture_path, filename)
                            dst_file = os.path.join(participant_path, filename)

                            # Dosyayı bir üst dizine (participant klasörüne) taşı
                            print(f"  Taşınıyor: {gesture_folder}\\{filename} -> {participant}\\{filename}")
                            shutil.move(src_file, dst_file)
                    
                    # Klasör boşaldıysa sil
                    if not os.listdir(gesture_path):
                        try:
                            os.rmdir(gesture_path)
                            print(f"  Boş klasör silindi: {gesture_folder}")
                        except OSError:
                            print(f"  UYARI: Klasör silinemedi (kilitli olabilir): {gesture_folder}")

if __name__ == "__main__":
    reorganize_files()
    print("\nDosya düzenleme işlemi tamamlandı.")