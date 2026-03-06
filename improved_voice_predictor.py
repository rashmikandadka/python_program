"""
Improved Voice Predictor with Better Audio Handling
UPDATED VERSION - Better calibrated thresholds to reduce false positives
"""

import numpy as np
import os
import librosa
import soundfile as sf
from audio_features import VoiceFeatureExtractor

class ImprovedVoicePredictor:
    """
    Enhanced voice predictor with robust audio preprocessing
    """
    
    def __init__(self, model_path='models/voice_model'):
        self.model_path = model_path
        self.feature_extractor = VoiceFeatureExtractor()
        
        # Load model components
        self.load_model()
        
        # Clinical thresholds (calibrated for real-world data)
        self.thresholds = {
            'jitter_high': 0.018,      # More lenient - only flag severe tremor
            'shimmer_high': 0.10,      # More lenient - only flag severe instability
            'hnr_low': 12,             # More lenient - 12 dB threshold
            'pitch_std_low': 8,        # More lenient - only flag very monotone speech
            'pitch_mean_low': 80,      # Very low pitch threshold
            'pitch_mean_high': 300     # Very high pitch threshold
        }
        
        print("✅ Improved Voice Predictor initialized")
    
    def load_model(self):
        """Load the trained model"""
        import joblib
        
        # Check if model exists
        if not os.path.exists(self.model_path):
            print(f"⚠️  Model not found at {self.model_path}")
            print("   Using threshold-based prediction only")
            self.model = None
            self.scaler = None
            self.model_type = 'threshold'
            return
        
        try:
            # Load scaler
            scaler_path = os.path.join(self.model_path, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                self.scaler = None
            
            # Load model type
            type_file = os.path.join(self.model_path, 'model_type.txt')
            if os.path.exists(type_file):
                with open(type_file, 'r') as f:
                    self.model_type = f.read().strip()
            else:
                self.model_type = 'random_forest'
            
            # Load model
            if self.model_type == 'neural_network':
                from tensorflow import keras
                model_file = os.path.join(self.model_path, 'voice_model.h5')
                if os.path.exists(model_file):
                    self.model = keras.models.load_model(model_file)
                else:
                    # Fallback to pkl
                    model_file = os.path.join(self.model_path, 'voice_model.pkl')
                    self.model = joblib.load(model_file)
                    self.model_type = 'random_forest'
            else:
                model_file = os.path.join(self.model_path, 'voice_model.pkl')
                self.model = joblib.load(model_file)
            
            print(f"✅ Model loaded: {self.model_type}")
            
        except Exception as e:
            print(f"⚠️  Error loading model: {e}")
            print("   Using threshold-based prediction only")
            self.model = None
            self.scaler = None
            self.model_type = 'threshold'
    
    def normalize_audio(self, audio_path, output_path=None):
        """
        Normalize audio file to standard format
        - Convert to 22050 Hz sample rate
        - Convert to mono
        - Normalize amplitude
        - Remove silence
        """
        try:
            print(f"[NORMALIZE] Processing: {audio_path}")
            
            # Load audio with librosa (handles multiple formats)
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            
            print(f"[NORMALIZE] Original - Duration: {len(y)/sr:.2f}s, SR: {sr}Hz")
            
            # Remove silence from beginning and end
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            if len(y_trimmed) < sr * 0.3:
                print("[NORMALIZE] Audio too short after trimming")
                return None
            
            # Normalize amplitude to -20 dB
            y_normalized = librosa.util.normalize(y_trimmed)
            
            # Apply gentle compression to reduce dynamic range
            y_compressed = np.sign(y_normalized) * np.log1p(np.abs(y_normalized) * 10) / np.log1p(10)
            
            print(f"[NORMALIZE] Normalized - Duration: {len(y_compressed)/sr:.2f}s")
            
            # Save normalized audio if output path provided
            if output_path:
                sf.write(output_path, y_compressed, sr)
                print(f"[NORMALIZE] Saved to: {output_path}")
                return output_path
            else:
                # Save to temporary file
                temp_path = audio_path.replace('.', '_normalized.')
                sf.write(temp_path, y_compressed, sr)
                print(f"[NORMALIZE] Saved to: {temp_path}")
                return temp_path
                
        except Exception as e:
            print(f"[NORMALIZE ERROR] {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def validate_audio(self, audio_path):
        """
        Validate audio file with improved checks
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            print(f"[VALIDATE] Duration: {duration:.2f}s, SR: {sr}Hz")
            
            # Check duration (0.5s to 120s)
            if duration < 0.5:
                return False, "⚠️  Audio too short (minimum 1 second required)"
            if duration > 120:
                return False, "⚠️  Audio too long (maximum 2 minutes)"
            
            # Check if mostly silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            if len(y_trimmed) < sr * 0.3:
                return False, "⚠️  Audio is mostly silence - please speak louder"
            
            # Check RMS energy
            rms = librosa.feature.rms(y=y)
            avg_rms = np.mean(rms)
            
            print(f"[VALIDATE] RMS Energy: {avg_rms:.6f}")
            
            if avg_rms < 0.0003:  # Very lenient threshold
                return False, "⚠️  Audio volume too low - please record closer to microphone"
            
            print("[VALIDATE] ✅ Validation passed")
            return True, "Valid audio"
            
        except Exception as e:
            print(f"[VALIDATE ERROR] {e}")
            return False, f"Error validating audio: {str(e)}"
    
    def predict(self, audio_path):
        """
        Predict with improved preprocessing
        """
        try:
            print(f"\n{'='*60}")
            print(f"IMPROVED VOICE PREDICTION")
            print(f"{'='*60}")
            
            # Step 1: Normalize audio
            print("\n[STEP 1] Normalizing audio...")
            normalized_path = self.normalize_audio(audio_path)
            
            if normalized_path is None:
                return "⚠️  Unable to process audio file", 0.0, None
            
            # Use normalized audio for further processing
            processing_path = normalized_path
            
            # Step 2: Validate
            print("\n[STEP 2] Validating audio...")
            is_valid, validation_msg = self.validate_audio(processing_path)
            
            if not is_valid:
                # Clean up temp file
                if normalized_path != audio_path and os.path.exists(normalized_path):
                    os.remove(normalized_path)
                return validation_msg, 0.0, None
            
            # Step 3: Extract features
            print("\n[STEP 3] Extracting features...")
            features = self.feature_extractor.extract_all_features(processing_path)
            
            if features is None:
                # Clean up temp file
                if normalized_path != audio_path and os.path.exists(normalized_path):
                    os.remove(normalized_path)
                return "⚠️  Unable to extract voice features", 0.0, None
            
            print(f"[SUCCESS] Extracted {len(features)} features")
            
            # Debug: Print key features
            print("\n[DEBUG] Key Features:")
            print(f"  Jitter: {features.get('jitter_local', 0):.4f}")
            print(f"  Shimmer: {features.get('shimmer_local', 0):.4f}")
            print(f"  HNR: {features.get('hnr', 0):.2f} dB")
            print(f"  Pitch Std: {features.get('pitch_std', 0):.2f} Hz")
            print(f"  Pitch Mean: {features.get('pitch_mean', 0):.2f} Hz")
            
            # Step 4: Calculate risk score using thresholds
            print("\n[STEP 4] Calculating risk score...")
            risk_score = self._calculate_risk_score(features)
            
            # Step 5: Use ML model if available
            ml_confidence = None
            if self.model is not None and self.scaler is not None:
                print("\n[STEP 5] Running ML model prediction...")
                try:
                    feature_array = self.feature_extractor.features_to_array(features)
                    feature_array_scaled = self.scaler.transform(feature_array)
                    
                    if self.model_type == 'neural_network':
                        ml_prob = float(self.model.predict(feature_array_scaled, verbose=0)[0][0])
                    else:
                        ml_prob = float(self.model.predict_proba(feature_array_scaled)[0][1])
                    
                    ml_confidence = ml_prob * 100
                    print(f"[ML MODEL] Parkinson's probability: {ml_confidence:.2f}%")
                    
                    # Combine threshold and ML predictions (weighted average)
                    combined_confidence = (risk_score * 0.4) + (ml_confidence * 0.6)
                    print(f"[COMBINED] Final confidence: {combined_confidence:.2f}%")
                    
                    # Use combined confidence
                    final_confidence = combined_confidence
                    
                except Exception as e:
                    print(f"[ML ERROR] {e}")
                    print("[FALLBACK] Using threshold-based prediction only")
                    final_confidence = risk_score
            else:
                print("[INFO] ML model not available, using threshold-based prediction")
                final_confidence = risk_score
            
            # Step 6: Make decision - ADJUSTED THRESHOLDS
            print("\n[STEP 6] Making final decision...")
            if final_confidence >= 65:  # Increased from 60
                result = "⚠️  Voice analysis indicates Parkinson's Disease"
            elif final_confidence >= 50:  # Increased from 40
                result = "⚠️  Voice analysis shows borderline indicators"
            else:
                result = "✅ Voice analysis indicates Healthy"
            
            # Step 7: Analyze features
            analysis = self._analyze_features(features)
            
            # Clean up temporary file
            if normalized_path != audio_path and os.path.exists(normalized_path):
                os.remove(normalized_path)
            
            print(f"\n[RESULT] {result}")
            print(f"[CONFIDENCE] {final_confidence:.2f}%")
            print(f"{'='*60}\n")
            
            return result, round(final_confidence, 2), analysis
            
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up temp file
            try:
                if 'normalized_path' in locals() and normalized_path != audio_path:
                    if os.path.exists(normalized_path):
                        os.remove(normalized_path)
            except:
                pass
            
            return "⚠️  Error during voice analysis", 0.0, None
    
    def _calculate_risk_score(self, features):
        """
        Calculate risk score from features - IMPROVED VERSION
        More conservative scoring to reduce false positives
        """
        risk_factors = 0
        total_factors = 100
        
        # Jitter (30 points) - Most important for Parkinson's
        jitter = features.get('jitter_local', 0)
        if jitter > self.thresholds['jitter_high']:
            risk_factors += 30
            print(f"  [+] High jitter: {jitter:.4f} (threshold: {self.thresholds['jitter_high']})")
        elif jitter > self.thresholds['jitter_high'] * 0.75:
            risk_factors += 15
            print(f"  [~] Borderline jitter: {jitter:.4f}")
        else:
            print(f"  [-] Normal jitter: {jitter:.4f}")
        
        # Shimmer (30 points) - Very important for Parkinson's
        shimmer = features.get('shimmer_local', 0)
        if shimmer > self.thresholds['shimmer_high']:
            risk_factors += 30
            print(f"  [+] High shimmer: {shimmer:.4f} (threshold: {self.thresholds['shimmer_high']})")
        elif shimmer > self.thresholds['shimmer_high'] * 0.75:
            risk_factors += 15
            print(f"  [~] Borderline shimmer: {shimmer:.4f}")
        else:
            print(f"  [-] Normal shimmer: {shimmer:.4f}")
        
        # HNR (20 points) - Important for voice quality
        hnr = features.get('hnr', 20)
        if hnr < self.thresholds['hnr_low'] and hnr > 0:
            risk_factors += 20
            print(f"  [+] Low HNR (breathy): {hnr:.2f} dB (threshold: {self.thresholds['hnr_low']})")
        elif hnr < self.thresholds['hnr_low'] * 1.3 and hnr > 0:
            risk_factors += 10
            print(f"  [~] Borderline HNR: {hnr:.2f} dB")
        else:
            print(f"  [-] Normal HNR: {hnr:.2f} dB")
        
        # Pitch variability (20 points) - Less weight, more variable in normal speech
        pitch_std = features.get('pitch_std', 20)
        pitch_mean = features.get('pitch_mean', 150)
        
        # Only penalize if VERY low AND mean pitch is reasonable
        if pitch_std < self.thresholds['pitch_std_low'] and 80 < pitch_mean < 300:
            risk_factors += 20
            print(f"  [+] Very monotone: pitch_std={pitch_std:.2f} Hz (threshold: {self.thresholds['pitch_std_low']})")
        elif pitch_std < self.thresholds['pitch_std_low'] * 1.5 and 80 < pitch_mean < 300:
            risk_factors += 8
            print(f"  [~] Somewhat monotone: pitch_std={pitch_std:.2f} Hz")
        else:
            print(f"  [-] Normal pitch variation: {pitch_std:.2f} Hz")
        
        risk_score = risk_factors
        print(f"\n  Risk Score: {risk_score}/{total_factors}")
        
        return risk_score
    
    def _analyze_features(self, features):
        """Provide detailed feature analysis"""
        analysis = {
            'pitch_variability': 'Normal',
            'voice_stability': 'Normal',
            'voice_quality': 'Normal',
            'key_indicators': []
        }
        
        jitter = features.get('jitter_local', 0)
        shimmer = features.get('shimmer_local', 0)
        hnr = features.get('hnr', 20)
        pitch_std = features.get('pitch_std', 20)
        pitch_mean = features.get('pitch_mean', 150)
        
        # Analyze each parameter
        if pitch_std < self.thresholds['pitch_std_low'] and 80 < pitch_mean < 300:
            analysis['pitch_variability'] = 'Reduced'
            analysis['key_indicators'].append(f'Monotone speech (pitch std: {pitch_std:.2f} Hz)')
        
        if jitter > self.thresholds['jitter_high']:
            analysis['voice_stability'] = 'Unstable'
            analysis['key_indicators'].append(f'Voice tremor detected (jitter: {jitter:.3f}%)')
        
        if shimmer > self.thresholds['shimmer_high']:
            analysis['voice_stability'] = 'Unstable'
            analysis['key_indicators'].append(f'Amplitude instability (shimmer: {shimmer:.3f}%)')
        
        if hnr < self.thresholds['hnr_low'] and hnr > 0:
            analysis['voice_quality'] = 'Breathy'
            analysis['key_indicators'].append(f'Breathy voice (HNR: {hnr:.1f} dB)')
        
        if not analysis['key_indicators']:
            analysis['key_indicators'].append('All voice parameters within normal ranges')
        
        return analysis


# Test function
if __name__ == "__main__":
    print("="*60)
    print("Improved Voice Predictor - Testing")
    print("="*60)
    
    predictor = ImprovedVoicePredictor('models/voice_model')
    
    test_audio = "test_voice.wav"
    if os.path.exists(test_audio):
        result, confidence, analysis = predictor.predict(test_audio)
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Result: {result}")
        print(f"Confidence: {confidence:.2f}%")
        
        if analysis:
            print("\n📊 Analysis:")
            print(f"  Pitch Variability: {analysis['pitch_variability']}")
            print(f"  Voice Stability: {analysis['voice_stability']}")
            print(f"  Voice Quality: {analysis['voice_quality']}")
            print("\n  Key Indicators:")
            for indicator in analysis['key_indicators']:
                print(f"    • {indicator}")
    else:
        print(f"\n⚠️  Test file not found: {test_audio}")