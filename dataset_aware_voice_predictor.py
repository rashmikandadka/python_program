"""
Dataset-Aware Voice Predictor for Parkinson's Detection
Handles both Figshare telephone recordings AND live recordings
"""

import numpy as np
import os
import librosa
import soundfile as sf
from audio_features import VoiceFeatureExtractor

class DatasetAwareVoicePredictor:
    """
    Predictor that adapts to different audio sources
    - Telephone recordings (Figshare dataset)
    - Live recordings (microphone/mobile)
    """
    
    def __init__(self, model_path='models/voice_model'):
        self.model_path = model_path
        self.feature_extractor = VoiceFeatureExtractor()
        self.load_model()
        
        # Separate thresholds for different recording types
        self.telephone_thresholds = {
            'jitter_high': 0.025,      # Higher for telephone quality
            'shimmer_high': 0.12,      # Higher for telephone compression
            'hnr_low': 10,             # Lower due to telephone noise
            'pitch_std_low': 6,        # Lower threshold for monotone
        }
        
        self.live_thresholds = {
            'jitter_high': 0.020,      # Stricter for clean recordings
            'shimmer_high': 0.10,      
            'hnr_low': 12,             
            'pitch_std_low': 8,        
        }
        
        print("✅ Dataset-Aware Voice Predictor initialized")
    
    def load_model(self):
        """Load ML model if available"""
        import joblib
        
        if not os.path.exists(self.model_path):
            print(f"⚠️  Model not found - using threshold-based prediction")
            self.model = None
            self.scaler = None
            self.model_type = 'threshold'
            return
        
        try:
            scaler_path = os.path.join(self.model_path, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            type_file = os.path.join(self.model_path, 'model_type.txt')
            if os.path.exists(type_file):
                with open(type_file, 'r') as f:
                    self.model_type = f.read().strip()
            else:
                self.model_type = 'random_forest'
            
            if self.model_type == 'neural_network':
                from tensorflow import keras
                model_file = os.path.join(self.model_path, 'voice_model.h5')
                self.model = keras.models.load_model(model_file)
            else:
                model_file = os.path.join(self.model_path, 'voice_model.pkl')
                self.model = joblib.load(model_file)
            
            print(f"✅ ML Model loaded: {self.model_type}")
        except Exception as e:
            print(f"⚠️  Model loading failed: {e}")
            self.model = None
            self.scaler = None
            self.model_type = 'threshold'
    
    def detect_recording_type(self, audio_path):
        """
        Detect if audio is telephone recording or live recording
        Returns: 'telephone' or 'live'
        """
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate spectral characteristics
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_centroid = np.mean(spectral_centroids)
            
            # Calculate bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            avg_bandwidth = np.mean(spectral_bandwidth)
            
            # Calculate energy distribution
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Check energy above 3400 Hz (telephone cutoff)
            high_freq_mask = freqs > 3400
            high_freq_energy = np.sum(S[high_freq_mask, :]) if np.any(high_freq_mask) else 0
            total_energy = np.sum(S)
            high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
            
            print(f"\n[DETECTION] Audio Analysis:")
            print(f"  Spectral Centroid: {avg_centroid:.2f} Hz")
            print(f"  Spectral Bandwidth: {avg_bandwidth:.2f} Hz")
            print(f"  High Freq Energy: {high_freq_ratio:.4f}")
            print(f"  Sample Rate: {sr} Hz")
            print(f"  Duration: {len(y)/sr:.2f}s")
            
            # Decision logic
            is_telephone = False
            
            # Telephone recordings have:
            # - Limited bandwidth (< 3400 Hz typically)
            # - Lower sample rates (8kHz or 16kHz)
            # - Very low high-frequency content
            
            if sr <= 16000:
                is_telephone = True
                print(f"  → Low sample rate detected (telephone)")
            elif avg_centroid < 1500:
                is_telephone = True
                print(f"  → Limited frequency range (telephone)")
            elif high_freq_ratio < 0.01:
                is_telephone = True
                print(f"  → Minimal high frequencies (telephone)")
            elif avg_bandwidth < 1000:
                is_telephone = True
                print(f"  → Narrow bandwidth (telephone)")
            
            recording_type = 'telephone' if is_telephone else 'live'
            print(f"  ✅ Detected as: {recording_type.upper()} recording")
            
            return recording_type
            
        except Exception as e:
            print(f"[DETECTION ERROR] {e}")
            return 'live'  # Default to live
    
    def normalize_audio(self, audio_path, recording_type):
        """
        Normalize audio based on recording type
        """
        try:
            print(f"\n[NORMALIZE] Processing {recording_type} recording...")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            
            # Remove silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            
            if len(y_trimmed) < sr * 0.3:
                return None
            
            # Different normalization for different types
            if recording_type == 'telephone':
                # More aggressive for telephone
                y_normalized = librosa.util.normalize(y_trimmed)
                # Boost amplitude more for telephone
                y_compressed = np.sign(y_normalized) * np.power(np.abs(y_normalized), 0.7)
            else:
                # Standard normalization for live
                y_normalized = librosa.util.normalize(y_trimmed)
                y_compressed = np.sign(y_normalized) * np.log1p(np.abs(y_normalized) * 10) / np.log1p(10)
            
            # Save to temp file
            temp_path = audio_path.replace('.', '_normalized.')
            sf.write(temp_path, y_compressed, sr)
            
            print(f"[NORMALIZE] ✅ Saved normalized audio")
            return temp_path
            
        except Exception as e:
            print(f"[NORMALIZE ERROR] {e}")
            return None
    
    def validate_audio(self, audio_path):
        """Validate audio file"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            if duration < 0.5:
                return False, "⚠️  Audio too short (minimum 1 second)"
            if duration > 120:
                return False, "⚠️  Audio too long (maximum 2 minutes)"
            
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            if len(y_trimmed) < sr * 0.3:
                return False, "⚠️  Audio is mostly silence"
            
            rms = librosa.feature.rms(y=y)
            if np.mean(rms) < 0.0002:
                return False, "⚠️  Audio volume too low"
            
            return True, "Valid audio"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def predict(self, audio_path):
        """
        Predict with automatic recording type detection
        """
        try:
            print(f"\n{'='*70}")
            print(f"DATASET-AWARE VOICE ANALYSIS")
            print(f"{'='*70}")
            
            # Step 1: Detect recording type
            recording_type = self.detect_recording_type(audio_path)
            
            # Select appropriate thresholds
            if recording_type == 'telephone':
                thresholds = self.telephone_thresholds
                print(f"\n📞 Using TELEPHONE thresholds (Figshare dataset compatible)")
            else:
                thresholds = self.live_thresholds
                print(f"\n🎤 Using LIVE recording thresholds")
            
            # Step 2: Normalize
            print(f"\n[STEP 1] Normalizing audio...")
            normalized_path = self.normalize_audio(audio_path, recording_type)
            
            if normalized_path is None:
                return "⚠️  Unable to process audio file", 0.0, None
            
            # Step 3: Validate
            print(f"\n[STEP 2] Validating audio...")
            is_valid, msg = self.validate_audio(normalized_path)
            if not is_valid:
                if normalized_path != audio_path:
                    os.remove(normalized_path)
                return msg, 0.0, None
            
            # Step 4: Extract features
            print(f"\n[STEP 3] Extracting features...")
            features = self.feature_extractor.extract_all_features(normalized_path)
            
            if features is None:
                if normalized_path != audio_path:
                    os.remove(normalized_path)
                return "⚠️  Unable to extract voice features", 0.0, None
            
            # Print key features
            print(f"\n[DEBUG] Key Features:")
            print(f"  Jitter: {features.get('jitter_local', 0):.4f}")
            print(f"  Shimmer: {features.get('shimmer_local', 0):.4f}")
            print(f"  HNR: {features.get('hnr', 0):.2f} dB")
            print(f"  Pitch Std: {features.get('pitch_std', 0):.2f} Hz")
            print(f"  Pitch Mean: {features.get('pitch_mean', 0):.2f} Hz")
            
            # Step 5: Calculate risk score
            print(f"\n[STEP 4] Calculating risk score...")
            risk_score = self._calculate_risk_score(features, thresholds, recording_type)
            
            # Step 6: Use ML model if available
            final_confidence = risk_score
            if self.model is not None and self.scaler is not None:
                print(f"\n[STEP 5] Running ML model...")
                try:
                    feature_array = self.feature_extractor.features_to_array(features)
                    feature_array_scaled = self.scaler.transform(feature_array)
                    
                    if self.model_type == 'neural_network':
                        ml_prob = float(self.model.predict(feature_array_scaled, verbose=0)[0][0])
                    else:
                        ml_prob = float(self.model.predict_proba(feature_array_scaled)[0][1])
                    
                    ml_confidence = ml_prob * 100
                    print(f"  ML Confidence: {ml_confidence:.2f}%")
                    
                    # Combine predictions
                    final_confidence = (risk_score * 0.5) + (ml_confidence * 0.5)
                    print(f"  Combined Confidence: {final_confidence:.2f}%")
                except Exception as e:
                    print(f"  ML Error: {e}, using threshold only")
            
            # Step 7: Make decision
            print(f"\n[STEP 6] Making final decision...")
            if final_confidence >= 65:
                result = "⚠️  Voice analysis indicates Parkinson's Disease"
            elif final_confidence >= 50:
                result = "⚠️  Voice analysis shows borderline indicators"
            else:
                result = "✅ Voice analysis indicates Healthy"
            
            # Analysis
            analysis = self._analyze_features(features, thresholds)
            
            # Cleanup
            if normalized_path != audio_path:
                os.remove(normalized_path)
            
            print(f"\n[RESULT] {result}")
            print(f"[CONFIDENCE] {final_confidence:.2f}%")
            print(f"[RECORDING TYPE] {recording_type.upper()}")
            print(f"{'='*70}\n")
            
            return result, round(final_confidence, 2), analysis
            
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return "⚠️  Error during analysis", 0.0, None
    
    def _calculate_risk_score(self, features, thresholds, recording_type):
        """Calculate risk score with recording-type-aware thresholds"""
        risk_factors = 0
        
        # Jitter (35 points) - MOST IMPORTANT
        jitter = features.get('jitter_local', 0)
        if jitter > thresholds['jitter_high']:
            risk_factors += 35
            print(f"  [+++] HIGH Jitter: {jitter:.4f} > {thresholds['jitter_high']}")
        elif jitter > thresholds['jitter_high'] * 0.7:
            risk_factors += 20
            print(f"  [++] Elevated Jitter: {jitter:.4f}")
        else:
            print(f"  [-] Normal Jitter: {jitter:.4f}")
        
        # Shimmer (35 points) - MOST IMPORTANT
        shimmer = features.get('shimmer_local', 0)
        if shimmer > thresholds['shimmer_high']:
            risk_factors += 35
            print(f"  [+++] HIGH Shimmer: {shimmer:.4f} > {thresholds['shimmer_high']}")
        elif shimmer > thresholds['shimmer_high'] * 0.7:
            risk_factors += 20
            print(f"  [++] Elevated Shimmer: {shimmer:.4f}")
        else:
            print(f"  [-] Normal Shimmer: {shimmer:.4f}")
        
        # HNR (20 points)
        hnr = features.get('hnr', 20)
        if 0 < hnr < thresholds['hnr_low']:
            risk_factors += 20
            print(f"  [+] Low HNR: {hnr:.2f} dB < {thresholds['hnr_low']}")
        elif 0 < hnr < thresholds['hnr_low'] * 1.3:
            risk_factors += 10
            print(f"  [~] Borderline HNR: {hnr:.2f} dB")
        else:
            print(f"  [-] Normal HNR: {hnr:.2f} dB")
        
        # Pitch Std (10 points) - LEAST WEIGHT
        pitch_std = features.get('pitch_std', 20)
        pitch_mean = features.get('pitch_mean', 150)
        
        if pitch_std < thresholds['pitch_std_low'] and 80 < pitch_mean < 300:
            risk_factors += 10
            print(f"  [+] Monotone: {pitch_std:.2f} Hz < {thresholds['pitch_std_low']}")
        else:
            print(f"  [-] Normal Pitch Variation: {pitch_std:.2f} Hz")
        
        print(f"\n  📊 Risk Score: {risk_factors}/100")
        print(f"  📱 Recording Type: {recording_type.upper()}")
        
        return risk_factors
    
    def _analyze_features(self, features, thresholds):
        """Analyze features and return structured results"""
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
        
        if jitter > thresholds['jitter_high']:
            analysis['voice_stability'] = 'Unstable'
            analysis['key_indicators'].append(f'Voice tremor (jitter: {jitter:.3f}%)')
        
        if shimmer > thresholds['shimmer_high']:
            analysis['voice_stability'] = 'Unstable'
            analysis['key_indicators'].append(f'Amplitude instability (shimmer: {shimmer:.3f}%)')
        
        if 0 < hnr < thresholds['hnr_low']:
            analysis['voice_quality'] = 'Breathy'
            analysis['key_indicators'].append(f'Breathy voice (HNR: {hnr:.1f} dB)')
        
        if pitch_std < thresholds['pitch_std_low']:
            analysis['pitch_variability'] = 'Reduced'
            analysis['key_indicators'].append(f'Monotone speech ({pitch_std:.2f} Hz)')
        
        if not analysis['key_indicators']:
            analysis['key_indicators'].append('All voice parameters within normal ranges')
        
        return analysis


# Test
if __name__ == "__main__":
    predictor = DatasetAwareVoicePredictor()
    
    test_file = "test_voice.wav"
    if os.path.exists(test_file):
        result, conf, analysis = predictor.predict(test_file)
        print(f"\nFinal Result: {result}")
        print(f"Confidence: {conf}%")