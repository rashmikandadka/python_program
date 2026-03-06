import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
from audio_features import VoiceFeatureExtractor

class VoiceModelTrainer:
    """
    Trains voice-based Parkinson's detection models.
    Supports both ML (Random Forest, SVM) and DL (Neural Network) approaches.
    """
    
    def __init__(self, model_type='neural_network'):
        """
        model_type: 'random_forest', 'svm', or 'neural_network'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = VoiceFeatureExtractor()
        
    def prepare_dataset(self, audio_folder, labels_csv=None):
        """
        Prepare dataset from audio files.
        
        audio_folder structure:
            audio_folder/
                healthy/
                    audio1.wav
                    audio2.wav
                parkinson/
                    audio1.wav
                    audio2.wav
        
        OR provide labels_csv with columns: filename, label (0=healthy, 1=parkinson)
        """
        features_list = []
        labels_list = []
        
        if labels_csv:
            # Load from CSV
            df = pd.read_csv(labels_csv)
            for idx, row in df.iterrows():
                audio_path = os.path.join(audio_folder, row['filename'])
                if os.path.exists(audio_path):
                    print(f"Processing: {row['filename']}")
                    features = self.feature_extractor.extract_all_features(audio_path)
                    if features:
                        feature_array = self.feature_extractor.features_to_array(features)
                        features_list.append(feature_array[0])
                        labels_list.append(row['label'])
        else:
            # Load from folder structure
            for class_name in ['healthy', 'parkinson']:
                class_folder = os.path.join(audio_folder, class_name)
                label = 0 if class_name == 'healthy' else 1
                
                if os.path.exists(class_folder):
                    for audio_file in os.listdir(class_folder):
                        if audio_file.endswith(('.wav', '.mp3', '.ogg')):
                            audio_path = os.path.join(class_folder, audio_file)
                            print(f"Processing: {audio_file}")
                            
                            features = self.feature_extractor.extract_all_features(audio_path)
                            if features:
                                feature_array = self.feature_extractor.features_to_array(features)
                                features_list.append(feature_array[0])
                                labels_list.append(label)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"\n✅ Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Healthy: {np.sum(y == 0)}, Parkinson: {np.sum(y == 1)}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2, save_path='models/voice_model'):
        """Train the selected model"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n🔥 Training {self.model_type} model...")
        
        if self.model_type == 'random_forest':
            self.model = self._train_random_forest(X_train_scaled, y_train)
            
        elif self.model_type == 'svm':
            self.model = self._train_svm(X_train_scaled, y_train)
            
        elif self.model_type == 'neural_network':
            self.model = self._train_neural_network(
                X_train_scaled, y_train, 
                X_test_scaled, y_test
            )
        
        # Evaluate
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✅ Model trained successfully!")
        print(f"📊 Test Accuracy: {accuracy * 100:.2f}%")
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Healthy', 'Parkinson']))
        
        # Save model
        self._save_model(save_path)
        
        return accuracy
    
    def _train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        return model
    
    def _train_svm(self, X_train, y_train):
        """Train SVM classifier"""
        model = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        return model
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train Neural Network"""
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=[early_stop],
            verbose=1
        )
        
        return model
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'neural_network':
            predictions = (self.model.predict(X_scaled) > 0.5).astype(int).flatten()
        else:
            predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == 'neural_network':
            proba = self.model.predict(X_scaled).flatten()
        else:
            proba = self.model.predict_proba(X_scaled)[:, 1]
        
        return proba
    
    def _save_model(self, save_path):
        """Save model and scaler"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(save_path, 'scaler.pkl'))
        
        # Save model
        if self.model_type == 'neural_network':
            self.model.save(os.path.join(save_path, 'voice_model.h5'))
        else:
            joblib.dump(self.model, os.path.join(save_path, 'voice_model.pkl'))
        
        # Save model type
        with open(os.path.join(save_path, 'model_type.txt'), 'w') as f:
            f.write(self.model_type)
        
        print(f"\n💾 Model saved to: {save_path}")
    
    def load_model(self, save_path):
        """Load saved model"""
        # Load model type
        with open(os.path.join(save_path, 'model_type.txt'), 'r') as f:
            self.model_type = f.read().strip()
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(save_path, 'scaler.pkl'))
        
        # Load model
        if self.model_type == 'neural_network':
            self.model = keras.models.load_model(
                os.path.join(save_path, 'voice_model.h5')
            )
        else:
            self.model = joblib.load(os.path.join(save_path, 'voice_model.pkl'))
        
        print(f"✅ Model loaded from: {save_path}")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Voice-Based Parkinson's Detection - Model Training")
    print("="*60)
    
    # Initialize trainer
    trainer = VoiceModelTrainer(model_type='neural_network')
    
    # Prepare dataset
    # Option 1: From folder structure
    X, y = trainer.prepare_dataset('voice_dataset/')
    
    # Option 2: From CSV
    # X, y = trainer.prepare_dataset('voice_data/', 'labels.csv')
    
    # Train model
    if len(X) > 0:
        trainer.train(X, y, save_path='models/voice_model')
    else:
        print("❌ No data found. Please prepare your dataset first.")