"""
train_comparison.py
Bu script, toplanan verileri yükler, özellik çıkarımı yapar ve
belirlenen birden fazla makine öğrenmesi modelini Grid Search ile eğiterek kıyaslar.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Opsiyonel kütüphaneler (Yüklü değilse atlanır)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("UYARI: XGBoost kütüphanesi bulunamadı. XGBoost eğitimi atlanacak.")

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
    print("UYARI: CatBoost kütüphanesi bulunamadı. CatBoost eğitimi atlanacak.")

def extract_features(time_series_data):
    """
    gesture_model.py ile aynı özellik çıkarımı.
    time_series_data shape: (time_steps, 34)
    Returns: 1D feature vector (170 features)
    """
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

def load_data():
    """
    deletedrows klasöründeki tüm katılımcı verilerini yükler ve pencereleme yapar.
    """
    X = []
    y = []
    print("Veriler yükleniyor ve özellikler çıkarılıyor...")
    
    # 6 katılımcı için döngü (p1...p6)
    for participant_id in range(1, 7):
        folder = f'deletedrows/p{participant_id}'
        if not os.path.exists(folder):
            continue
        
        files = glob.glob(f'{folder}/*/*.npy')
        for file in files:
            basename = os.path.basename(file)
            # Dosya adı formatı: gesturename_timestamp.npy
            gesture_name = basename.split('_')[0]
            
            data = np.load(file)
            
            # Pencereleme Ayarları (gesture_model.py ile uyumlu)
            # 200Hz örnekleme hızı varsayımıyla:
            window_size = 20  # 100ms
            stride = 10       # 50ms ( %50 örtüşme)
            
            if len(data) < window_size:
                continue
                
            # Sliding Window
            num_windows = (len(data) - window_size) // stride + 1
            for i in range(num_windows):
                start = i * stride
                end = start + window_size
                window = data[start:end]
                
                feat = extract_features(window)
                X.append(feat)
                y.append(gesture_name)
                
    return np.array(X), np.array(y)

def main():
    # 1. Veriyi Hazırla
    X, y = load_data()
    print(f"Toplam Örnek Sayısı: {len(X)}")
    
    if len(X) == 0:
        print("HATA: Hiç veri bulunamadı! 'deletedrows' klasörünü kontrol edin.")
        return

    # Etiketleri Sayısala Çevir (XGBoost/CatBoost için gerekli)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    print(f"Sınıflar ({len(classes)}): {classes}")
    
    # Train/Test Split (%80 Train, %20 Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Ölçeklendirme (Scaling)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 2. Modelleri ve Parametreleri Tanımla
    models_config = {
        "SVM": {
            "model": SVC(random_state=42),
            "params": {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        }
    }
    
    # Eğer kütüphaneler varsa listeye ekle
    if HAS_XGB:
        models_config["XGBoost"] = {
            "model": XGBClassifier(eval_metric='mlogloss', random_state=42),
            "params": {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        }
        
    if HAS_CAT:
        models_config["CatBoost"] = {
            "model": CatBoostClassifier(verbose=0, random_state=42),
            "params": {
                'iterations': [50, 100, 150, 200], # n_estimators ile aynı
                'depth': [4, 6, 10],
                'learning_rate': [0.01, 0.1]
            }
        }

    # 3. Eğitim ve Karşılaştırma Döngüsü
    results_summary = []
    date_str = datetime.now().strftime("%d%m%Y_%H%M")
    os.makedirs("model_reports", exist_ok=True) # Raporlar için klasör
    
    print(f"\n=== {len(models_config)} FARKLI MODEL İÇİN GRID SEARCH BAŞLIYOR ===")
    
    for name, config in models_config.items():
        print(f"\n>>> Model İşleniyor: {name}")
        print(f"    Parametreler: {config['params']}")
        
        # Grid Search
        grid = GridSearchCV(
            estimator=config['model'], 
            param_grid=config['params'], 
            cv=5, 
            n_jobs=-1, 
            verbose=1, 
            scoring='accuracy'
        )
        grid.fit(X_train, y_train)
        
        # En iyi modeli değerlendir
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"    ✓ En İyi Parametreler: {grid.best_params_}")
        print(f"    ✓ Test Doğruluğu: %{acc*100:.2f}")
        
        results_summary.append({
            "Model": name,
            "Accuracy": acc,
            "Best Params": grid.best_params_
        })
        
        # Raporu Kaydet
        filename = f"model_reports/{name.replace(' ', '')}_{date_str}"
        with open(f"{filename}.txt", "w", encoding="utf-8") as f:
            f.write(f"MODEL: {name}\n")
            f.write(f"Tarih: {date_str}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Best Params: {grid.best_params_}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred, target_names=classes))
            
        # Confusion Matrix Çiz
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {name} (Acc: %{acc*100:.2f})')
        plt.ylabel('Gerçek Etiket')
        plt.xlabel('Tahmin Edilen')
        plt.tight_layout()
        plt.savefig(f"{filename}_cm.png")
        plt.close()

    # 4. Genel Karşılaştırma Raporu
    summary_file = f"model_reports/GENEL_KARSILASTIRMA_{date_str}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"MODEL KARŞILAŞTIRMA RAPORU - {date_str}\n")
        f.write("="*100 + "\n")
        f.write(f"{'Model Adı':<20} | {'Doğruluk':<10} | {'En İyi Parametreler'}\n")
        f.write("-" * 100 + "\n")
        
        # Doğruluğa göre sırala (Büyükten küçüğe)
        results_summary.sort(key=lambda x: x['Accuracy'], reverse=True)
        
        for res in results_summary:
            f.write(f"{res['Model']:<20} | %{res['Accuracy']*100:>8.2f} | {res['Best Params']}\n")
            
        f.write("\n" + "="*100 + "\n")
        f.write(f"KAZANAN MODEL: {results_summary[0]['Model']}\n")
            
    print(f"\nEğitim tamamlandı! Tüm raporlar 'model_reports' klasörüne kaydedildi.")
    print(f"Genel özet: {summary_file}")

if __name__ == "__main__":
    main()
