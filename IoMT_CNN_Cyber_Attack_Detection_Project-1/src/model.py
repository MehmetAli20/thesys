import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_autoencoder_lstm_model(input_dim, num_classes):
    # GPU kullanımını optimize et
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU optimizasyonu etkinleştirildi: {len(gpus)} GPU bulundu")
        except RuntimeError as e:
            print(f"⚠️ GPU ayarı hatası: {e}")
    
    inputs = keras.Input(shape=(input_dim,))
    
    # Veri girdisine Gaussian gürültü ekle
    x = layers.GaussianNoise(0.02)(inputs)
    
    # Encoder kısmı
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Reshape for LSTM
    reshape_dim = 32
    sequence_len = 4
    
    # Reshape: -> (batch_size, sequence_len, reshape_dim)
    x = layers.Dense(reshape_dim * sequence_len, activation='relu')(x)
    x = layers.Reshape((sequence_len, reshape_dim))(x)
    
    # LSTM katmanları
    x = layers.LSTM(64, dropout=0.3, return_sequences=True)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.LSTM(32, dropout=0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Decoder kısmı
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Çıkış katmanı
    output = layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=output)
    
    # Model derleme
    optimizer = keras.optimizers.Adam(
        learning_rate=0.0005,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    # Metrikler - hem binary hem multi-class için tutarlı
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy',
        metrics=metrics
    )
    
    print("✅ Model oluşturuldu (TensorFlow 2.10.0 Uyumlu)")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            mode='min'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            mode='min',
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]
    
    print(f"🚀 Eğitim başlatılıyor: {len(X_train)} örnek, batch_size={batch_size}")
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history