import tensorflow as tf
import numpy as np

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check if GPU is available
print("\nGPU Available:", tf.config.list_physical_devices('GPU'))

# Create a simple model and run it on GPU
if tf.config.list_physical_devices('GPU'):
    print("\nRunning test computation on GPU...")
    with tf.device('/GPU:0'):
        # Create some random data
        x = tf.random.normal([1000, 1000])
        y = tf.random.normal([1000, 1000])
        
        # Perform matrix multiplication
        start_time = tf.timestamp()
        z = tf.matmul(x, y)
        end_time = tf.timestamp()
        
        print(f"Computation time: {end_time - start_time:.4f} seconds")
        print("Test completed successfully!")
else:
    print("\nNo GPU found. Please check your CUDA and cuDNN installation.") 