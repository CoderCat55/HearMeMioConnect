import os
from collections import defaultdict

def dosya_say_ve_ozetle(ana_dizin):
    if not os.path.exists(ana_dizin):
        print(f"Hata: '{ana_dizin}' klasörü bulunamadı!")
        return

    # Kelime bazlı toplamları tutmak için bir sözlük
    toplam_sozluk = defaultdict(int)

    print(f"{'PARTICIPANT':<12} | {'KELİME':<12} | {'SAYI'}")
    print("-" * 40)

    # Katılımcı klasörlerini gez (p1, p2...)
    katilimcilar = sorted([d for d in os.listdir(ana_dizin) if os.path.isdir(os.path.join(ana_dizin, d))])

    for p in katilimcilar:
        p_yolu = os.path.join(ana_dizin, p)
        kelimeler = sorted([k for k in os.listdir(p_yolu) if os.path.isdir(os.path.join(p_yolu, k))])
        
        for kelime in kelimeler:
            kelime_yolu = os.path.join(p_yolu, kelime)
            sayi = len([f for f in os.listdir(kelime_yolu) if os.path.isfile(os.path.join(kelime_yolu, f))])
            
            # Ekrana yazdır ve toplama ekle
            print(f"{p:<12} | {kelime:<12} | {sayi}")
            toplam_sozluk[kelime] += sayi

    # Genel Toplam Bölümü
    print("\n" + "="*40)
    print(f"{'KELİME':<12} | {'GENEL TOPLAM'}")
    print("-" * 40)
    
    # Kelimeleri alfabetik olarak sıralayıp toplamları yazdır
    for kelime in sorted(toplam_sozluk.keys()):
        print(f"{kelime:<12} | {toplam_sozluk[kelime]}")

# Çalıştır
dosya_say_ve_ozetle(r"C:\Users\Bemol\Documents\GitHub\HearMeMioConnect\twodelete")