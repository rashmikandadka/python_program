import numpy as np
import os
import joblib
from tensorflow import keras
from audio_features import VoiceFeatureExtractor

class VoicePredictor:
    """
    Predicts Parkinson's disease from voice recordings.
    """
    
    def __init__(self, model_path='models/voice_model'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.model_type = None
        self.feature_extractor = VoiceFeatureExtractor()
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        # Load model type
        type_file = os.path.join(self.model_path, 'model_type.txt')
        if os.path.exists(type_file):
            with open(type_file, 'r') as f:
                self.model_type = f.read().strip()
        else:
            self.model_type = 'random_forest'  # Default to random_forest
        
        # Load scaler
        scaler_path = os.path.join(self.model_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Load model
        if self.model_type == 'neural_network':
            model_file = os.path.join(self.model_path, 'voice_model.h5')
            if os.path.exists(model_file):
                self.model = keras.models.load_model(model_file)
            else:
                # Try .pkl file instead
                model_file = os.path.join(self.model_path, 'voice_model.pkl')
                self.model = joblib.load(model_file)
                self.model_type = 'random_forest'
        else:
            model_file = os.path.join(self.model_path, 'voice_model.pkl')
            self.model = joblib.load(model_file)
        
        print(f"✅ Voice model loaded: {self.model_type}")
    
    def validate_audio(self, audio_path):
        """
        Validate if the audio file is suitable for analysis.
        RELAXED validation - more lenient with uploaded files
        Returns: (is_valid, error_message)
        """
        try:
            import librosa
            
            print(f"[DEBUG] Validating audio: {audio_path}")
            
            # Try to load the audio
            y, sr = librosa.load(audio_path, sr=None)
            
            print(f"[DEBUG] Audio loaded - Duration: {len(y)/sr:.2f}s, Sample rate: {sr}Hz")
            
            # Check duration - RELAXED (accept 0.5s to 120s)
            duration = len(y) / sr
            
            if duration < 0.5:
                return False, "⚠️ Audio too short. Please record at least 1-2 seconds"
            
            if duration > 120:
                return False, "⚠️ Audio too long. Please keep it under 2 minutes"
            
            # Check if audio is mostly silence - RELAXED
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)  # More lenient (was 20)
            
            if len(y_trimmed) < sr * 0.3:  # At least 0.3 seconds (was 0.5)
                print("[DEBUG] Audio is mostly silence")
                return False, "⚠️ Audio is mostly silence. Please speak clearly and loudly"
            
            # Check if it's too noisy - RELAXED
            rms = librosa.feature.rms(y=y)
            avg_rms = np.mean(rms)
            
            print(f"[DEBUG] Average RMS: {avg_rms}")
            
            if avg_rms < 0.0005:  # More lenient (was 0.001)
                return False, "⚠️ Audio volume too low. Please speak louder or adjust microphone"
            
            print("[DEBUG] Audio validation PASSED")
            return True, "Valid audio"
            
        except Exception as e:
            print(f"[DEBUG] Validation error: {str(e)}")
            return False, f"Error validating audio: {str(e)}"
    
    def predict(self, audio_path):
        """
        Predict Parkinson's from voice recording.
        Returns: (result_text, confidence, feature_analysis)
        """
        try:
            print(f"\n{'='*60}")
            print(f"VOICE PREDICTION STARTED")
            print(f"{'='*60}")
            print(f"Audio file: {audio_path}")
            
            # Step 1: Validate audio
            is_valid, validation_msg = self.validate_audio(audio_path)
            if not is_valid:
                print(f"[ERROR] Validation failed: {validation_msg}")
                return validation_msg, 0.0, None
            
            # Step 2: Extract features
            print("\n[STEP 1] Extracting voice features...")
            features = self.feature_extractor.extract_all_features(audio_path)
            
            if features is None:
                print("[ERROR] Feature extraction failed")
                return "⚠️ Unable to analyze audio. Please ensure clear speech with minimal background noise", 0.0, None
            
            print(f"[SUCCESS] Extracted {len(features)} features")
            
            # Debug: Print some feature values
            print("\n[DEBUG] Sample features:")
            for key, value in list(features.items())[:5]:
                print(f"  {key}: {value:.4f}")
            
            # Step 3: Prepare for prediction
            print("\n[STEP 2] Preparing features for prediction...")
            feature_array = self.feature_extractor.features_to_array(features)
            print(f"[DEBUG] Feature array shape: {feature_array.shape}")
            
            feature_array_scaled = self.scaler.transform(feature_array)
            print(f"[DEBUG] Scaled feature array shape: {feature_array_scaled.shape}")
            
            # Step 4: Predict
            print("\n[STEP 3] Making prediction...")
            if self.model_type == 'neural_network':
                prediction_proba = float(self.model.predict(feature_array_scaled, verbose=0)[0][0])
            else:
                prediction_proba = float(self.model.predict_proba(feature_array_scaled)[0][1])
            
            print(f"[DEBUG] Raw prediction probability: {prediction_proba}")
            
            # Step 5: Interpret results
            if prediction_proba >= 0.5:
                result = "⚠️ Voice analysis indicates Parkinson's Disease"
                confidence = prediction_proba * 100
            else:
                result = "✅ Voice analysis indicates Healthy"
                confidence = (1 - prediction_proba) * 100
            
            print(f"[RESULT] {result}")
            print(f"[CONFIDENCE] {confidence:.2f}%")
            
            # Step 6: Analyze key features
            feature_analysis = self._analyze_features(features)
            
            print(f"\n{'='*60}")
            print(f"PREDICTION COMPLETED")
            print(f"{'='*60}\n")
            
            return result, round(confidence, 2), feature_analysis
            
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "⚠️ Error during voice analysis. Please try again with a different recording", 0.0, None
    
    def _analyze_features(self, features):
        """
        Analyze key voice features and provide clinical insights.
        """
        analysis = {
            'pitch_variability': 'Normal',
            'voice_stability': 'Normal',
            'voice_quality': 'Normal',
            'key_indicators': []
        }
        
        # Pitch variability (reduced in Parkinson's)
        pitch_std = features.get('pitch_std', 20)
        if pitch_std < 15:
            analysis['pitch_variability'] = 'Reduced'
            analysis['key_indicators'].append('Monotone speech (reduced pitch variation)')
        elif pitch_std > 40:
            analysis['pitch_variability'] = 'High'
            analysis['key_indicators'].append('Excessive pitch variation')
        
        # Jitter (increased in Parkinson's)
        jitter = features.get('jitter_local', 0)
        if jitter > 0.01:
            analysis['voice_stability'] = 'Unstable'
            analysis['key_indicators'].append('Voice tremor detected (high jitter)')
        
        # Shimmer (increased in Parkinson's)
        shimmer = features.get('shimmer_local', 0)
        if shimmer > 0.05:
            analysis['voice_stability'] = 'Unstable'
            analysis['key_indicators'].append('Amplitude variations (high shimmer)')
        
        # HNR (reduced in Parkinson's - breathier voice)
        hnr = features.get('hnr', 20)
        if hnr < 15:
            analysis['voice_quality'] = 'Breathy'
            analysis['key_indicators'].append('Breathy voice quality (low HNR)')
        
        if not analysis['key_indicators']:
            analysis['key_indicators'].append('All voice parameters within normal range')
        
        return analysis


def get_recording_instructions():
    """
    Returns instructions for recording voice samples.
    """
    return {
        'task_options': [
            {
                'name': 'Sustained Vowel',
                'instruction': 'Say "Aaaaaaah" for 5-10 seconds at a comfortable pitch',
                'best_for': 'Detecting tremor and voice instability'
            },
            {
                'name': 'Reading Passage',
                'instruction': 'Read: "The sun shines brightly in the clear blue sky"',
                'best_for': 'Natural speech analysis'
            },
            {
                'name': 'Counting',
                'instruction': 'Count from 1 to 20 at normal speed',
                'best_for': 'Rhythm and articulation assessment'
            }
        ],
        'recording_tips': [
            '🎤 Use a good quality microphone in a quiet room',
            '📱 Hold phone/mic about 6 inches from your mouth',
            '📊 Speak at your normal volume (not too loud or soft)',
            '⏱️ Record for at least 3-5 seconds',
            '🔇 Minimize background noise'
        ],
        'important_notes': [
            '⚠️ This is a screening tool, not a diagnostic test',
            '👨‍⚕️ Consult a neurologist for proper diagnosis',
            '📊 Multiple recordings improve accuracy'
        ]
    }


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Voice-Based Parkinson's Detection - Prediction")
    print("="*60)
    
    # Test prediction
    predictor = VoicePredictor('models/voice_model')
    
    test_audio = "test_voice.wav"
    if os.path.exists(test_audio):
        result, confidence, analysis = predictor.predict(test_audio)
        
        print(f"\n{result}")
        print(f"Confidence: {confidence:.2f}%")
        
        if analysis:
            print("\n📊 Feature Analysis:")
            print(f"  Pitch Variability: {analysis['pitch_variability']}")
            print(f"  Voice Stability: {analysis['voice_stability']}")
            print(f"  Voice Quality: {analysis['voice_quality']}")
            print("\n  Key Indicators:")
            for indicator in analysis['key_indicators']:
                print(f"    • {indicator}")