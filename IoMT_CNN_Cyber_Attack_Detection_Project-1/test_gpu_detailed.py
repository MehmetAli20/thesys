import tensorflow as tf
import numpy as np
import os

print("="*50)
print("TensorFlow GPU Test")
print("="*50)

# TensorFlow versiyonu
print("\n1. TensorFlow Version:", tf.__version__)

# GPU cihazlarını kontrol et
gpus = tf.config.list_physical_devices('GPU')
print("\n2. Available GPUs:", gpus)

# CUDA ve cuDNN bilgilerini göster
print("\n3. CUDA and cuDNN Information:")
print("CUDA built:", tf.test.is_built_with_cuda())
print("GPU available:", tf.test.is_gpu_available())
print("GPU device name:", tf.test.gpu_device_name())

# GPU bellek büyümesini etkinleştir
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n4. GPU Memory Growth: Enabled")
    except RuntimeError as e:
        print("\n4. GPU Memory Growth Error:", e)

# Test hesaplaması
if gpus:
    print("\n5. Running GPU Test Computation...")
    with tf.device('/GPU:0'):
        # Büyük matris oluştur
        matrix_size = 2000
        print(f"Creating {matrix_size}x{matrix_size} matrices...")
        
        # CPU'da oluştur
        start_time = tf.timestamp()
        x = tf.random.normal([matrix_size, matrix_size])
        y = tf.random.normal([matrix_size, matrix_size])
        
        # GPU'da hesapla
        print("Performing matrix multiplication on GPU...")
        z = tf.matmul(x, y)
        
        # Sonucu bekle
        _ = z.numpy()
        end_time = tf.timestamp()
        
        print(f"Computation time: {end_time - start_time:.4f} seconds")
        print("GPU Test completed successfully!")
else:
    print("\n5. No GPU found. Please check your CUDA and cuDNN installation.")

print("\n" + "="*50) 