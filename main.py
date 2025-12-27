"""handles multiprocessing and user commands"""
import multiprocessing as mp
from multiprocessing import Process, shared_memory
import numpy as np
from model import GestureClassifier
import time

# Constants
STREAM_BUFFER_SIZE = 1000  # ~5 seconds at 200Hz
CALIBRATION_BUFFER_SIZE = 600  # 3 seconds at 200Hz

CALIBRATION_DURATION = 3  # seconds
CLASSIFICATION_DURATION = 3  # same!

CALIBRATION_STARTS = 5
CLASSIFICATION_STARTS = 5

def data_acquisition_process(stream_mem_name, calib_mem_name, stream_index, 
                            calib_index, recording_flag, recording_gesture):
    """Process 1: Continuously acquires data from Myo armbands"""
    from mioconnect.src.myodriver import MyoDriver
    from mioconnect.src.config import Config
    
    # Attach to shared memory
    shm_stream = shared_memory.SharedMemory(name=stream_mem_name)
    stream_buffer = np.ndarray((STREAM_BUFFER_SIZE, 34), dtype=np.float32, buffer=shm_stream.buf)
    
    shm_calib = shared_memory.SharedMemory(name=calib_mem_name)
    calib_buffer = np.ndarray((CALIBRATION_BUFFER_SIZE, 34), dtype=np.float32, buffer=shm_calib.buf)
    
    # Initialize MyoDriver with ALL parameters
    config = Config()
    myo_driver = MyoDriver(
        config, 
        stream_buffer=stream_buffer,
        stream_index=stream_index,
        calib_buffer=calib_buffer,
        calib_index=calib_index,
        recording_flag=recording_flag,
        recording_gesture=recording_gesture
    )
    
    # Connect to Myos
    myo_driver.run()  # Now returns after connections complete!
    # Verify both Myos have device names
    if len(myo_driver.myos) >= 2:
        for i, myo in enumerate(myo_driver.myos):
            if myo.device_name is None:
                print(f"✗ Warning: Myo {i} (connection {myo.connection_id}) has no device name!")
            else:
                print(f"✓ Myo {i}: {myo.device_name} (connection {myo.connection_id})")
    print("Data acquisition process ready!")
    
    # Run forever - NOW this works correctly
    while True:
        myo_driver.receive()
        
def get_calibration_buffer_from_shared_mem(calib_buffer, calib_index):
    """Read recorded calibration data"""
    # Return only the filled portion of the buffer
    num_samples = calib_index.value
    if num_samples == 0:
        return None
    return calib_buffer[:num_samples].copy()

def get_recent_data_from_shared_mem(stream_buffer, stream_index, window_seconds=CLASSIFICATION_DURATION):
    """Read the last N seconds from the streaming buffer"""
    # Calculate how many samples we need
    samples_per_second = 100  # Approximate (depends on actual rate)
    num_samples = int(window_seconds * samples_per_second)
    
    # Get current position
    current_idx = stream_index.value
    
    # Handle wrap-around for circular buffer
    if current_idx < num_samples:
        # Buffer hasn't filled yet
        return stream_buffer[:current_idx].copy()
    else:
        # Calculate wrapped indices
        buffer_size = len(stream_buffer)
        start_idx_wrapped = (current_idx - num_samples) % buffer_size
        current_idx_wrapped = current_idx % buffer_size
        
        # Check if we need to wrap around
        if start_idx_wrapped < current_idx_wrapped:
            # No wrap-around needed
            return stream_buffer[start_idx_wrapped:current_idx_wrapped].copy()
        else:
            # Wrap-around: get two parts and concatenate in correct order
            part1 = stream_buffer[start_idx_wrapped:]  # From start to end of buffer
            part2 = stream_buffer[:current_idx_wrapped]  # From beginning to current
            return np.concatenate([part1, part2])
        
def Calibrate(gesture_name, calib_buffer, calib_index, recording_flag, 
              recording_gesture, classifier):
    """Called from main process when user wants to calibrate"""
    """print(f"Calibration will start in ", end='', flush=True)
    for i in range(CALIBRATION_STARTS, 0, -1):
        print(f"{i}... ", end='', flush=True)
        time.sleep(1)
    print("\n")
    print(f"Recording calibration for '{gesture_name}' - '{CALIBRATION_DURATION} seconds...")
    """ 
    #şimdilik veri toplayacağumuz için burası kapalı.
    # Reset calibration buffer
    calib_index.value = 0
    
    # Set flag in shared memory
    gesture_bytes = gesture_name.encode('utf-8')
    for i, byte in enumerate(gesture_bytes[:50]):  # Max 50 chars
        recording_gesture[i] = byte
    recording_flag.value = 1  # Start recording
    
    # Wait 3 seconds
    print("Recording... ", end='', flush=True)
    for i in range(CALIBRATION_DURATION):
        time.sleep(1)
        print(f"{CALIBRATION_DURATION-i}... ", end='', flush=True)
    print("Done!")
    
    # Stop recording
    recording_flag.value = 0
    time.sleep(0.5)  # Let final batch flush
    # Read recorded data
    recorded_data = get_calibration_buffer_from_shared_mem(calib_buffer, calib_index)
    
    if recorded_data is None or len(recorded_data) == 0:
        print("ERROR: No data was recorded! Check if Myos are connected.")
        return
    
    # Add to classifier
    classifier.add_calibration_sample(gesture_name, recorded_data)
    
    # Save to disk
    import os
    os.makedirs('calibration_data', exist_ok=True)
    timestamp = int(time.time())
    np.save(f'calibration_data/{gesture_name}_{timestamp}.npy', recorded_data)
    
    print(f"Calibration complete! Saved {len(recorded_data)} samples")

def Classify(stream_buffer, stream_index, classifier):
    """Called from main process when user wants to classify"""
    time.sleep(CLASSIFICATION_DURATION)
    # Read current data from shared memory (last 1 second)
    current_data = get_recent_data_from_shared_mem(stream_buffer, stream_index, window_seconds=CLASSIFICATION_DURATION)
    
    if current_data is None or len(current_data) < 10:
        print("ERROR: Not enough data to classify!")
        return None
    
    # Extract features
    features = GestureClassifier.extract_features(current_data)
    
    # Classify
    result = classifier.classify(features)
    print(f"Predicted gesture: {result}")
    return result

def LiveClassify():
    """Continuously classify gestures in real-time."""
    if classifier.model is None:
        print("ERROR: Model not trained yet! Please train the model first using 'tr'.")
        return

    print("\n>>> Starting LIVE classification... classify fo 20 times <<<")
    for i in range(1,20,1):
        print(f"\nClassification {i}/20:")
        Classify(stream_buffer, stream_index, classifier)
        print("waiting1 sec for other classification")
        time.sleep(CLASSIFICATION_DURATION)

def Train(classifier):
    """Called from main process when user wants to train"""
    print("Training model...")
    success = classifier.train()
    if success:
        classifier.save_model('model.pkl')
        print("Training complete!")
    else:
        print("Training failed! Make sure you have calibration data.")

def RealTimePipeline(stream_buffer, stream_index):
    """
    Yeni Eklenen Fonksiyon:
    Model 1 (Segmentasyon) ve Model 2 (SVM) kullanarak canlı sınıflandırma yapar.
    """
    print("\n=== Real-Time Pipeline (Model 1 + Model 2) ===")
    
    # Gerekli kütüphaneleri burada import ediyoruz ki dosyanın başı karışmasın
    try:
        from model1 import RestDetector
        from model2 import GestureClassifierSVM
        import pickle
        import os
    except ImportError as e:
        print(f"Hata: Gerekli model dosyaları (model1.py, model2.py) bulunamadı: {e}")
        return

    # 1. Model 2'yi (SVM) Yükle
    model_path = "model2.pkl"
    if not os.path.exists(model_path):
        print("HATA: model2.pkl bulunamadı. Lütfen önce model2.py'yi çalıştırıp eğitin.")
        return

    classifier = GestureClassifierSVM()
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            classifier.svm = data['model']
            classifier.scaler = data['scaler']
            classifier.is_trained = True
        print("Model 2 (SVM) başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return

    # 2. Model 1'i (RestDetector) Başlat ve Kalibre Et
    # Ortam gürültüsünü öğrenmek için kısa bir kalibrasyon yapıyoruz
    detector = RestDetector(window_size=20, threshold_factor=3.5)
    
    print("\n--- HIZLI KALİBRASYON (REST) ---")
    print("Lütfen kolunuzu 2 saniye boyunca serbest/hareketsiz bırakın...")
    time.sleep(1.0)
    print("Veri okunuyor...")
    time.sleep(2.0)
    
    # Son 2 saniyelik veriyi al
    rest_data = get_recent_data_from_shared_mem(stream_buffer, stream_index, window_seconds=2.0)
    
    if rest_data is None or len(rest_data) < 50:
        print("Hata: Kalibrasyon için yeterli veri okunamadı. Myo bağlı mı?")
        return
        
    detector.fit([rest_data])
    print(f"Kalibrasyon Tamam. Gürültü Eşiği: {detector.threshold:.4f}")
    print("Sistem Başladı! (Çıkmak için CTRL+C)")

    # Dinamik Veri Toplama Parametreleri
    POLL_INTERVAL = 0.1         # Döngü hızı (saniye)
    MAX_GESTURE_DURATION = 3.0  # Maksimum hareket süresi (bundan uzunsa keser)
    SILENCE_TIMEOUT = 0.5       # Hareket bitti kabul etmek için gereken sessizlik süresi

    try:
        while True:
            # 1. Adım: Rest Kontrolü (Küçük pencere ~0.2sn)
            window_data = get_recent_data_from_shared_mem(stream_buffer, stream_index, window_seconds=0.2)
            
            if window_data is None or len(window_data) < 20:
                time.sleep(0.05)
                continue
            
            # Eğer Rest ise başa dön
            if detector.predict(window_data):
                time.sleep(0.05)
                continue
            
            # 2. Adım: Hareket Algılandı -> Dinamik Veri Toplama
            print("\n>>> HAREKET ALGILANDI! Veri toplanıyor...", end="", flush=True)
            
            collected_data = [window_data] # Başlangıcı ekle
            start_time = time.time()
            silence_start = None
            
            # Hareketi içeren son geçerli parçanın indeksi (Başlangıçta 1 parça var)
            valid_data_end_index = 1 
            
            while (time.time() - start_time) < MAX_GESTURE_DURATION:
                time.sleep(POLL_INTERVAL)
                
                # Yeni gelen parçayı al
                chunk = get_recent_data_from_shared_mem(stream_buffer, stream_index, window_seconds=POLL_INTERVAL)
                
                if chunk is not None and len(chunk) > 0:
                    collected_data.append(chunk)
                    
                    # Bu parça Rest mi?
                    is_rest = detector.predict(chunk)
                    if is_rest:
                        if silence_start is None:
                            silence_start = time.time()
                        elif (time.time() - silence_start) > SILENCE_TIMEOUT:
                            print(" Bitti (Rest).")
                            break # İç döngüden çık
                    else:
                        silence_start = None # Hareket devam ediyor
                        print(".", end="", flush=True)
                        # Aktif veri geldiği için geçerli son indeksi güncelle
                        valid_data_end_index = len(collected_data)
            
            # 3. Adım: Sınıflandırma
            # Sadece geçerli (aktif) kısımları alıyoruz, sondaki rest kısmını atıyoruz
            final_data = collected_data[:valid_data_end_index]
            
            if final_data:
                full_gesture = np.vstack(final_data)
                
                # Çok kısa hareketleri filtrele (Gürültü olabilir)
                if len(full_gesture) > 40: # ~0.4 saniye altı gürültüdür
                    prediction = classifier.predict(full_gesture)
                    print(f"\n>>> SONUÇ: {prediction} (Süre: {len(full_gesture)/100:.2f}s)")
                else:
                    print("\n(Çok kısa hareket, atlandı)")
            
            # Tekrar tetiklenmemesi için kısa bekleme
            time.sleep(0.5)
            print("Hazır...")
            
    except KeyboardInterrupt:
        print("\nReal-Time Pipeline durduruldu. Ana menüye dönülüyor.")

def Command(stream_buffer, stream_index, calib_buffer, calib_index, 
           recording_flag, recording_gesture, classifier):
    value = input("Enter your command majesty: ")
    match value:
        case "tr":  #train
            print("now will run train function")
            Train(classifier)
        case "cf": # classify <3
            print("now will run classify function")
            print(f"Classify will start in ", end='', flush=True)
            for i in range(CLASSIFICATION_STARTS, 0, -1):
                print(f"{i}... ", end='', flush=True)
                time.sleep(1)
            print("\n")
            print("Classifying gesture...")
            Classify(stream_buffer, stream_index, classifier)
        case "cb":  #calibrate
            print("now will run calibrate function")
            gesture_name = input("Which gesture would you like to calibrate? ")
            Calibrate(gesture_name, calib_buffer, calib_index, recording_flag, 
                     recording_gesture, classifier)
        case "live": # live classification
            print("now will run live classify function")
            print(f"Classify will start in ", end='', flush=True)
            for i in range(CLASSIFICATION_STARTS, 0, -1):
                print(f"{i}... ", end='', flush=True)
                time.sleep(1)
            print("\n")
            print("Classifying gesture...")
            LiveClassify()
        case "rt": # Real-Time Pipeline (Yeni Komut)
            print("Real-Time Pipeline başlatılıyor...")
            RealTimePipeline(stream_buffer, stream_index)
        case _:
            print("Invalid command! Use: tr, cf, cb, live, or rt")

if __name__ == "__main__":
    print("=== Gesture Recognition System ===")
    print("Initializing...")
    
    # Create shared memory buffers
    shm_stream = shared_memory.SharedMemory(create=True, size=STREAM_BUFFER_SIZE*34*4)
    stream_buffer = np.ndarray((STREAM_BUFFER_SIZE, 34), dtype=np.float32, buffer=shm_stream.buf)
    stream_buffer.fill(0)  # Initialize to zero
    
    shm_calib = shared_memory.SharedMemory(create=True, size=CALIBRATION_BUFFER_SIZE*34*4)
    calib_buffer = np.ndarray((CALIBRATION_BUFFER_SIZE, 34), dtype=np.float32, buffer=shm_calib.buf)
    calib_buffer.fill(0)
    
    # Create shared indices and flags
    stream_index = mp.Value('i', 0)
    calib_index = mp.Value('i', 0)
    recording_flag = mp.Value('i', 0)
    recording_gesture = mp.Array('c', 50)
    
    # Start data acquisition process
    data_process = Process(
        target=data_acquisition_process,
        args=(shm_stream.name, shm_calib.name, stream_index, calib_index, 
              recording_flag, recording_gesture)
    )
    data_process.daemon = True  # Dies when main process dies
    data_process.start()
    time.sleep(2)  # wait for dataprocess to start
    # Give it time to connect
    print("Connecting to Myo armbands...")
    time.sleep(5)

    if not data_process.is_alive():
        print("ERROR: Data acquisition process failed to start!")
        print("Check if Myo dongle is connected and MyoConnect is closed")
    
    # Give it time to connect
    print("Connecting to Myo armbands...")
    time.sleep(5)
    
    # Initialize classifier in main process
    classifier = GestureClassifier()
    classifier.load_calibration_data()
    
    print("\nSystem ready! Available commands: tr= train, cf =classify, cb= calibrate, rt= realtime")
    print()
    
    # Command loop
    try:
        while True:
            Command(stream_buffer, stream_index, calib_buffer, calib_index,
                   recording_flag, recording_gesture, classifier)
    except KeyboardInterrupt:
        print("\nShutting down...")
        data_process.terminate()
        data_process.join()
        shm_stream.close()
        shm_stream.unlink()
        shm_calib.close()
        shm_calib.unlink()
        print("Goodbye!")
