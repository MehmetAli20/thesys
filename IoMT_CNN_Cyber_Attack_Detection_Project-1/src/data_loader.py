import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import numpy as np
import os

# Define attack categories
ATTACK_CATEGORIES_19 = { 
    'ARP_Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT-DDoS-Connect_Flood',
    'MQTT-DDoS-Publish_Flood': 'MQTT-DDoS-Publish_Flood',
    'MQTT-DoS-Connect_Flood': 'MQTT-DoS-Connect_Flood',
    'MQTT-DoS-Publish_Flood': 'MQTT-DoS-Publish_Flood',
    'MQTT-Malformed_Data': 'MQTT-Malformed_Data',
    'Recon-OS_Scan': 'Recon-OS_Scan',
    'Recon-Ping_Sweep': 'Recon-Ping_Sweep',
    'Recon-Port_Scan': 'Recon-Port_Scan',
    'Recon-VulScan': 'Recon-VulScan',
    'TCP_IP-DDoS-ICMP': 'DDoS-ICMP',
    'TCP_IP-DDoS-SYN': 'DDoS-SYN',
    'TCP_IP-DDoS-TCP': 'DDoS-TCP',
    'TCP_IP-DDoS-UDP': 'DDoS-UDP',
    'TCP_IP-DoS-ICMP': 'DoS-ICMP',
    'TCP_IP-DoS-SYN': 'DoS-SYN',
    'TCP_IP-DoS-TCP': 'DoS-TCP',
    'TCP_IP-DoS-UDP': 'DoS-UDP',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_6 = {  
    'Spoofing': 'Spoofing',
    'MQTT-DDoS-Connect_Flood': 'MQTT',
    'MQTT-DDoS-Publish_Flood': 'MQTT',
    'MQTT-DoS-Connect_Flood': 'MQTT',
    'MQTT-DoS-Publish_Flood': 'MQTT',
    'MQTT-Malformed_Data': 'MQTT',
    'Recon-OS_Scan': 'Recon',
    'Recon-Ping_Sweep': 'Recon',
    'Recon-Port_Scan': 'Recon',
    'Recon-VulScan': 'Recon',
    'DDoS-ICMP': 'DDoS',
    'DDoS-SYN': 'DDoS',
    'DDoS-TCP': 'DDoS',
    'DDoS-UDP': 'DDoS',
    'DoS-ICMP': 'DoS',
    'DoS-SYN': 'DoS',
    'DoS-TCP': 'DoS',
    'DoS-UDP': 'DoS',
    'Benign': 'Benign'
}

ATTACK_CATEGORIES_2 = {  
    'ARP_Spoofing': 'attack',
    'MQTT-DDoS-Connect_Flood': 'attack',
    'MQTT-DDoS-Publish_Flood': 'attack',
    'MQTT-DoS-Connect_Flood': 'attack',
    'MQTT-DoS-Publish_Flood': 'attack',
    'MQTT-Malformed_Data': 'attack',
    'Recon-OS_Scan': 'attack',
    'Recon-Ping_Sweep': 'attack',
    'Recon-Port_Scan': 'attack',
    'Recon-VulScan': 'attack',
    'TCP_IP-DDoS-ICMP': 'attack',
    'TCP_IP-DDoS-SYN': 'attack',
    'TCP_IP-DDoS-TCP': 'attack',
    'TCP_IP-DDoS-UDP': 'attack',
    'TCP_IP-DoS-ICMP': 'attack',
    'TCP_IP-DoS-SYN': 'attack',
    'TCP_IP-DoS-TCP': 'attack',
    'TCP_IP-DoS-UDP': 'attack',
    'Benign': 'Benign'
}

def get_attack_category(file_name, class_config): 
    """Get attack category from file name."""

    if class_config == 2:
        categories = ATTACK_CATEGORIES_2
    elif class_config == 6:
        categories = ATTACK_CATEGORIES_6
    else:  # Default to 19 classes 
        categories = ATTACK_CATEGORIES_19  

    for key in categories:
        if key in file_name:
            return categories[key]
        
def balance_dataset(df, target_column, n_samples=None):
    """
    Her sınıftan eşit sayıda örnek alarak veri setini dengeler.
    n_samples belirtilmezse en az örneğe sahip sınıftaki örnek sayısı kullanılır.
    """
    # Her sınıftaki örnek sayısını say
    class_counts = df[target_column].value_counts()
    
    # Eğer n_samples belirtilmemişse, en az örneğe sahip sınıfın örnek sayısını kullan
    if n_samples is None:
        n_samples = class_counts.min()
    
    # Her sınıftan n_samples kadar rastgele örnek al
    balanced_dfs = []
    for class_label in class_counts.index:
        class_df = df[df[target_column] == class_label]
        sampled_df = class_df.sample(n=min(n_samples, len(class_df)), random_state=42)
        balanced_dfs.append(sampled_df)
    
    # Tüm örnekleri birleştir
    balanced_df = pd.concat(balanced_dfs, axis=0)
    
    # Verileri karıştır
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

def load_and_preprocess_data(data_dir, class_config=2, max_samples=200000):
    """
    Veri setini yükler, dengeler ve ön işleme yapar.
    """
    print("📝 Veri yükleme işlemi başlatı...")
    
    # Tüm CSV dosyalarını birleştir
    all_files = []
    for filename in os.listdir(os.path.join(data_dir, 'train')):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_dir, 'train', filename)
            df = pd.read_csv(file_path)
            
            # Dosya adından saldırı türünü belirle
            attack_type = get_attack_category(filename, class_config)
            
            # Etiket sütunu ekle
            df['label'] = attack_type
            
            # Temel temizlik
            df = df.drop_duplicates()
            df = df.dropna()
            
            all_files.append(df)
            print(f"  ✓ {filename} yüklendi: {len(df)} kayıt")
    
    # Tüm verileri tek bir DataFrame'de birleştir
    combined_df = pd.concat(all_files, axis=0, ignore_index=True)
    print(f"📊 Toplam veri sayısı: {len(combined_df)}")
    
    # Sınıf yapılandırmasına göre etiketleri ayarla
    if class_config == 2:
        combined_df['label'] = combined_df['label'].apply(lambda x: 'Attack' if x != 'Benign' else 'Benign')
    elif class_config == 6:
        combined_df['label'] = combined_df['label'].apply(lambda x: ATTACK_CATEGORIES_6.get(x, x))
    
    # Veri dengesizliğini kontrol et ve raporla
    class_distribution = combined_df['label'].value_counts()
    print("\n📊 Sınıf dağılımı:")
    for label, count in class_distribution.items():
        print(f"  • {label}: {count} ({count/len(combined_df)*100:.1f}%)")
    
    # Her sınıf için maksimum örnek sayısını belirle
    max_samples = min(max_samples, class_distribution.min())
    
    # Her sınıftan rastgele örnekleme yap
    balanced_dfs = []
    for label in combined_df['label'].unique():
        class_df = combined_df[combined_df['label'] == label]
        if len(class_df) > max_samples:
            sampled_df = class_df.sample(n=max_samples, random_state=42)
        else:
            sampled_df = class_df
        balanced_dfs.append(sampled_df)
    
    # Dengeli veri setini oluştur
    balanced_df = pd.concat(balanced_dfs, axis=0, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Özellikleri ve etiketleri ayır
    X = balanced_df.drop('label', axis=1)
    y = balanced_df['label']
    
    # Verileri normalize et
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ Veriler normalize edildi (StandardScaler)")
    
    # Etiketleri sayısal değerlere dönüştür
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Verileri eğitim (%60), doğrulama (%20) ve test (%20) setlerine böl
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded, 
        test_size=0.4,
        random_state=42,
        stratify=y_encoded
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=0.5, 
        random_state=42,
        stratify=y_temp
    )
    
    print(f"\n✅ Veri bölümleme tamamlandı (60/20/20):")
    print(f"  • Eğitim seti: {X_train.shape[0]} örnek")
    print(f"  • Doğrulama seti: {X_val.shape[0]} örnek")
    print(f"  • Test seti: {X_test.shape[0]} örnek")
    
    # Etiketleri kategorik formata dönüştür
    from tensorflow.keras.utils import to_categorical
    num_classes = len(label_encoder.classes_)
    y_train_categorical = to_categorical(y_train, num_classes)
    y_val_categorical = to_categorical(y_val, num_classes)
    y_test_categorical = to_categorical(y_test, num_classes)
    
    return X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, label_encoder
