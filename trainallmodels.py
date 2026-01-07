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
from sklearn.model_selection import GridSearchCV

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
        # 1. Organize and Load Data
        self.organize_files(DATA_DIRECTORY)
        X, y, gesture_map = self._load_nested_data(DATA_DIRECTORY)
        if X is None: return

        # 2. Split and Scale
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # 3. Define the Grid Search
        date_str = datetime.now().strftime("%d%m%Y")
        rf_report_filename = f"rfhyperparameter3{date_str}.txt"
        best_model_filename = f"rfbestmodel{date_str}" # CM ve Metrikler için
        
        param_grid = {
            'n_estimators': list(range(100, 201, 5)),
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False]
        }

        print(f"\nStarting RF Grid Search (saving to {rf_report_filename})...")
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                    cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
        grid_search.fit(X_train, y_train)

        # --- EN İYİ MODEL DEĞERLENDİRMESİ ---
        self.model = grid_search.best_estimator_
        y_pred = self.model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=list(gesture_map.keys()))
        cm = confusion_matrix(y_test, y_pred)

        # 4. Save results to Text File (Hyperparameter Report)
        with open(rf_report_filename, "w", encoding="utf-8") as f:
            f.write(f"RF HYPERPARAMETER REPORT - {datetime.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write("="*100 + "\n")
            f.write(f"{'Rank':<5} | {'Mean Acc':<10} | {'Parameters'}\n")
            f.write("-" * 100 + "\n")

            results = grid_search.cv_results_
            for i in range(len(results['params'])):
                rank = results['rank_test_score'][i]
                mean_acc = results['mean_test_score'][i]
                params = results['params'][i]
                f.write(f"{rank:<5} | {mean_acc*100:>8.2f}% | {params}\n")
            
            # DOSYA ALTINA EN İYİ MODELİ EKLEME
            f.write("\n" + "="*100 + "\n")
            f.write(f"EN IYI MODEL: {grid_search.best_params_}\n")
            f.write(f"Test Seti Accuracy: {acc*100:.2f}%\n")
            f.write("="*100 + "\n")

        # 5. Save Best Model Metrics (RF Best Model File)
        with open(f"{best_model_filename}.txt", "w", encoding="utf-8") as f:
            f.write(f"BEST MODEL EVALUATION - {date_str}\n")
            f.write(f"Best Params: {grid_search.best_params_}\n\n")
            f.write(f"Accuracy Score: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\nConfusion Matrix:\n")
            f.write(np.array2string(cm))

        # 6. Plot & Save Confusion Matrix (Görsel)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=gesture_map.keys(), yticklabels=gesture_map.keys())
        plt.title(f'Confusion Matrix - {best_model_filename}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{best_model_filename}_cm.png")
        print(f"Confusion Matrix kaydedildi: {best_model_filename}_cm.png")

        print("\nBest Model ist: ", grid_search.best_estimator_)
        self.plot_rf_heatmap(grid_search)
    def plot_rf_heatmap(self, grid_search):
        """Visualizes how n_estimators and max_depth affect accuracy."""
        results_df = pd.DataFrame(grid_search.cv_results_)

        # Fill 'None' in max_depth with a string so it shows up in the plot
        results_df['param_max_depth'] = results_df['param_max_depth'].fillna('None')

        # Create pivot table for the heatmap
        # Note: This averages scores across other parameters (like min_samples_split)
        viz_data = results_df.pivot_table(index='param_max_depth', 
                                          columns='param_n_estimators', 
                                          values='mean_test_score')

        plt.figure(figsize=(10, 7))
        sns.heatmap(viz_data, annot=True, cmap='YlGnBu', fmt='.4f')
        
        plt.title('Random Forest Accuracy: Max Depth vs. N Estimators')
        plt.xlabel('N Estimators (Number of Trees)')
        plt.ylabel('Max Depth (Tree Complexity)')
        plt.tight_layout()
        plt.savefig('rf_hyperparameter_heatmap3.png')
        print("Heatmap saved as 'rf_hyperparameter_heatmap3.png'")
        plt.show()
if __name__ == "__main__":
    clf = GestureClassifier()
    clf.run_full_training()