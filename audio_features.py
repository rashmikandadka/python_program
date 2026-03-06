"""
Voice Feature Extraction for Parkinson's Detection
Extracts acoustic features from voice recordings
"""

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import os

class VoiceFeatureExtractor:
    """
    Extracts voice features relevant to Parkinson's detection:
    - Pitch (F0) statistics
    - Jitter (voice stability)
    - Shimmer (amplitude variation)
    - Harmonics-to-Noise Ratio (HNR)
    - MFCCs (voice quality)
    """
    
    def __init__(self, sr=22050):
        self.sr = sr
    
    def extract_all_features(self, audio_path):
        """
        Extract all features from an audio file.
        Returns: dictionary of features
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)
            
            # Remove silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            if len(y_trimmed) < sr * 0.5:  # At least 0.5 seconds
                print(f"Warning: Audio too short after trimming")
                return None
            
            # Extract features
            features = {}
            
            # 1. Pitch features using Parselmouth (Praat)
            pitch_features = self._extract_pitch_features(audio_path)
            features.update(pitch_features)
            
            # 2. Jitter and Shimmer (voice stability)
            stability_features = self._extract_stability_features(audio_path)
            features.update(stability_features)
            
            # 3. Harmonics-to-Noise Ratio
            features['hnr'] = self._extract_hnr(audio_path)
            
            # 4. MFCCs (Mel-Frequency Cepstral Coefficients)
            mfcc_features = self._extract_mfcc_features(y_trimmed, sr)
            features.update(mfcc_features)
            
            # 5. Spectral features
            spectral_features = self._extract_spectral_features(y_trimmed, sr)
            features.update(spectral_features)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def _extract_pitch_features(self, audio_path):
        """Extract pitch-related features using Praat"""
        try:
            sound = parselmouth.Sound(audio_path)
            pitch = call(sound, "To Pitch", 0.0, 75, 500)
            
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values != 0]  # Remove unvoiced frames
            
            if len(pitch_values) == 0:
                return {
                    'pitch_mean': 0,
                    'pitch_std': 0,
                    'pitch_min': 0,
                    'pitch_max': 0
                }
            
            return {
                'pitch_mean': np.mean(pitch_values),
                'pitch_std': np.std(pitch_values),
                'pitch_min': np.min(pitch_values),
                'pitch_max': np.max(pitch_values)
            }
        except Exception as e:
            print(f"Pitch extraction error: {e}")
            return {'pitch_mean': 0, 'pitch_std': 0, 'pitch_min': 0, 'pitch_max': 0}
    
    def _extract_stability_features(self, audio_path):
        """Extract jitter and shimmer (voice stability indicators)"""
        try:
            sound = parselmouth.Sound(audio_path)
            
            # Create PointProcess for jitter/shimmer
            pitch = call(sound, "To Pitch", 0.0, 75, 500)
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
            
            # Jitter (local)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            
            # Shimmer (local)
            shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            return {
                'jitter_local': jitter_local if not np.isnan(jitter_local) else 0,
                'shimmer_local': shimmer_local if not np.isnan(shimmer_local) else 0
            }
        except Exception as e:
            print(f"Stability features error: {e}")
            return {'jitter_local': 0, 'shimmer_local': 0}
    
    def _extract_hnr(self, audio_path):
        """Extract Harmonics-to-Noise Ratio (voice quality)"""
        try:
            sound = parselmouth.Sound(audio_path)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            return hnr if not np.isnan(hnr) else 0
        except Exception as e:
            print(f"HNR extraction error: {e}")
            return 0
    
    def _extract_mfcc_features(self, y, sr):
        """Extract MFCC features (voice quality characteristics)"""
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            return {
                f'mfcc_{i}_mean': np.mean(mfccs[i]) for i in range(13)
            }
        except Exception as e:
            print(f"MFCC extraction error: {e}")
            return {f'mfcc_{i}_mean': 0 for i in range(13)}
    
    def _extract_spectral_features(self, y, sr):
        """Extract spectral features"""
        try:
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            return {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'zcr_mean': np.mean(zcr)
            }
        except Exception as e:
            print(f"Spectral features error: {e}")
            return {
                'spectral_centroid_mean': 0,
                'spectral_rolloff_mean': 0,
                'zcr_mean': 0
            }
    
    def features_to_array(self, features):
        """
        Convert feature dictionary to numpy array for ML model.
        Maintains consistent feature order.
        """
        feature_order = [
            'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max',
            'jitter_local', 'shimmer_local', 'hnr',
            'spectral_centroid_mean', 'spectral_rolloff_mean', 'zcr_mean'
        ] + [f'mfcc_{i}_mean' for i in range(13)]
        
        feature_array = np.array([[features.get(key, 0) for key in feature_order]])
        return feature_array


# Test the extractor
if __name__ == "__main__":
    print("="*60)
    print("Voice Feature Extractor - Test")
    print("="*60)
    
    extractor = VoiceFeatureExtractor()
    
    # Test with a sample audio file
    test_file = "test_audio.wav"
    
    if os.path.exists(test_file):
        print(f"\nTesting with: {test_file}")
        features = extractor.extract_all_features(test_file)
        
        if features:
            print("\n✅ Features extracted successfully!")
            print(f"\nSample features:")
            for key, value in list(features.items())[:10]:
                print(f"  {key}: {value:.4f}")
        else:
            print("\n❌ Feature extraction failed")
    else:
        print(f"\n⚠️ Test file not found: {test_file}")
        print("Place a .wav file as 'test_audio.wav' to test")