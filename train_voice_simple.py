"""
Simplified Voice Model Training Script
Train a model to detect Parkinson's from voice recordings
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from audio_features import VoiceFeatureExtractor

def prepare_dataset(dataset_folder):
    """
    Load and extract features from voice dataset
    
    Expected folder structure:
        dataset_folder/
            healthy/
                audio1.wav
                audio2.wav
            parkinson/
                audio1.wav
                audio2.wav
    """
    print("="*60)
    print("STEP 1: Loading Voice Dataset")
    print("="*60)
    
    extractor = VoiceFeatureExtractor()
    features_list = []
    labels_list = []
    
    # Process healthy samples
    healthy_folder = os.path.join(dataset_folder, 'healthy')
    if os.path.exists(healthy_folder):
        print(f"\n📁 Processing healthy samples from: {healthy_folder}")
        healthy_files = [f for f in os.listdir(healthy_folder) if f.endswith('.wav')]
        print(f"   Found {len(healthy_files)} files")
        
        for i, audio_file in enumerate(healthy_files, 1):
            audio_path = os.path.join(healthy_folder, audio_file)
            print(f"   [{i}/{len(healthy_files)}] Processing: {audio_file}")
            
            features = extractor.extract_all_features(audio_path)
            if features:
                feature_array = extractor.features_to_array(features)
                features_list.append(feature_array[0])
                labels_list.append(0)  # 0 = healthy
    else:
        print(f"⚠️ Warning: Healthy folder not found at {healthy_folder}")
    
    # Process Parkinson samples
    parkinson_folder = os.path.join(dataset_folder, 'parkinson')
    if os.path.exists(parkinson_folder):
        print(f"\n📁 Processing Parkinson samples from: {parkinson_folder}")
        parkinson_files = [f for f in os.listdir(parkinson_folder) if f.endswith('.wav')]
        print(f"   Found {len(parkinson_files)} files")
        
        for i, audio_file in enumerate(parkinson_files, 1):
            audio_path = os.path.join(parkinson_folder, audio_file)
            print(f"   [{i}/{len(parkinson_files)}] Processing: {audio_file}")
            
            features = extractor.extract_all_features(audio_path)
            if features:
                feature_array = extractor.features_to_array(features)
                features_list.append(feature_array[0])
                labels_list.append(1)  # 1 = parkinson
    else:
        print(f"⚠️ Warning: Parkinson folder not found at {parkinson_folder}")
    
    if len(features_list) == 0:
        print("\n❌ ERROR: No features extracted!")
        print("\nPlease check:")
        print("1. Your dataset folder structure is correct")
        print("2. Audio files are in .wav format")
        print("3. Files are not corrupted")
        return None, None
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"\n✅ Dataset prepared successfully!")
    print(f"   Total samples: {X.shape[0]}")
    print(f"   Features per sample: {X.shape[1]}")
    print(f"   Healthy samples: {np.sum(y == 0)}")
    print(f"   Parkinson samples: {np.sum(y == 1)}")
    
    return X, y


def train_model(X, y, save_path='models/voice_model'):
    """
    Train a Random Forest classifier
    """
    print("\n" + "="*60)
    print("STEP 2: Training Voice Model")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Scale features
    print(f"\n⚙️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print(f"\n🌲 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print(f"\n📈 Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model trained successfully!")
    print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")
    
    print(f"\n📋 Detailed Results:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Healthy', 'Parkinson']))
    
    # Save model
    print(f"\n💾 Saving model to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(save_path, 'scaler.pkl'))
    print(f"   ✓ Saved scaler.pkl")
    
    # Save model
    joblib.dump(model, os.path.join(save_path, 'voice_model.pkl'))
    print(f"   ✓ Saved voice_model.pkl")
    
    # Save model type
    with open(os.path.join(save_path, 'model_type.txt'), 'w') as f:
        f.write('random_forest')
    print(f"   ✓ Saved model_type.txt")
    
    print(f"\n✅ Model saved successfully!")
    
    return model, scaler, accuracy


def main():
    """Main training function"""
    print("\n" + "="*70)
    print("  🎤 PARKINSON'S VOICE DETECTION - MODEL TRAINING")
    print("="*70)
    
    # Check if dataset exists
    dataset_folder = 'voice_dataset'
    
    if not os.path.exists(dataset_folder):
        print(f"\n❌ ERROR: Dataset folder not found!")
        print(f"\nPlease create the following structure:")
        print(f"   {dataset_folder}/")
        print(f"   ├── healthy/")
        print(f"   │   ├── audio1.wav")
        print(f"   │   └── audio2.wav")
        print(f"   └── parkinson/")
        print(f"       ├── audio1.wav")
        print(f"       └── audio2.wav")
        return
    
    # Step 1: Prepare dataset
    X, y = prepare_dataset(dataset_folder)
    
    if X is None:
        return
    
    # Check if we have enough samples
    if len(X) < 10:
        print(f"\n⚠️ Warning: Only {len(X)} samples found!")
        print(f"   Recommended: At least 50 samples (25 per class)")
        print(f"   Continuing anyway...")
    
    # Step 2: Train model
    model, scaler, accuracy = train_model(X, y)
    
    # Final summary
    print("\n" + "="*70)
    print("  ✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   • Model Type: Random Forest")
    print(f"   • Accuracy: {accuracy * 100:.2f}%")
    print(f"   • Samples Used: {len(X)}")
    print(f"   • Model Location: models/voice_model/")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Run the Flask app: python app.py")
    print(f"   2. Open browser: http://localhost:5000")
    print(f"   3. Test with voice recordings!")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()