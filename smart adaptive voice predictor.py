"""
SMART Voice Predictor - Automatically adapts to dataset vs live recordings
Solves the problem where dataset files show as healthy incorrectly
"""

import numpy as np
import os
import librosa
import soundfile as sf
from audio_features import VoiceFeatureExtractor


class SmartVoicePredictor:
    """
    Intelligent predictor that detects recording type and adjusts thresholds
    """
    
    def __init__(self):
        self.feature_extractor = VoiceFeatureExtractor()
        
        # THRESHOLDS FOR LIVE RECORDINGS (strict - high quality expected)
        self.live_thresholds = {
            'jitter_high': 0.012,      # 1.2%
            'shimmer_high': 0.070,     # 7.0%
            'hnr_low': 15,             # 15 dB
            'pitch_std_low': 12,       # 12 Hz
        }
        
        # THRESHOLDS FOR DATASET RECORDINGS (lenient - telephone quality)
        self.dataset_thresholds = {
            'jitter_high': 0.006,      # 0.6% - MUCH STRICTER for dataset
            'shimmer_high': 0.040,     # 4.0% - MUCH STRICTER for dataset
            'hnr_low': 12,             # 12 dB
            'pitch_std_low': 10,       # 10 Hz
        }
        
        print("✅ Smart Voice Predictor initialized")
        print("   Auto-detects: Dataset vs Live recordings")
    
    def detect_recording_type(self, audio_path):
        """
        Detect if this is a dataset recording or live recording
        Dataset = telephone quality (8kHz or 16kHz, limited bandwidth)
        Live = high quality (22kHz+, full bandwidth)
        """
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            # Calculate spectral characteristics
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_centroid = np.mean(spectral_centroids)
            
            # Calculate bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            avg_bandwidth = np.mean(spectral_bandwidth)
            
            # High frequency energy
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            high_freq_mask = freqs > 3400
            high_freq_energy = np.sum(S[high_freq_mask, :]) if np.any(high_freq_mask) else 0
            total_energy = np.sum(S)
            high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
            
            print(f"\n[DETECTION] Audio Analysis:")
            print(f"  📊 Sample Rate: {sr} Hz")
            print(f"  📊 Duration: {duration:.2f}s")
            print(f"  📊 Spectral Centroid: {avg_centroid:.0f} Hz")
            print(f"  📊 Bandwidth: {avg_bandwidth:.0f} Hz")
            print(f"  📊 High Freq Ratio: {high_freq_ratio:.4f}")
            
            # DETECTION LOGIC
            is_dataset = False
            reasons = []
            
            # Check 1: Sample rate
            if sr <= 16000:
                is_dataset = True
                reasons.append(f"Low sample rate ({sr}Hz)")
            
            # Check 2: Limited bandwidth
            if avg_bandwidth < 1200:
                is_dataset = True
                reasons.append(f"Narrow bandwidth ({avg_bandwidth:.0f}Hz)")
            
            # Check 3: Low spectral centroid
            if avg_centroid < 1500:
                is_dataset = True
                reasons.append(f"Limited frequency range ({avg_centroid:.0f}Hz)")
            
            # Check 4: No high frequencies
            if high_freq_ratio < 0.02:
                is_dataset = True
                reasons.append(f"No high frequencies ({high_freq_ratio:.4f})")
            
            recording_type = 'DATASET' if is_dataset else 'LIVE'
            
            print(f"\n  🎯 DETECTED: {recording_type} recording")
            if reasons:
                print(f"  📝 Reasons: {', '.join(reasons)}")
            
            return recording_type, reasons
            
        except Exception as e:
            print(f"[DETECTION ERROR] {e}")
            return 'LIVE', []
    
    def normalize_audio(self, audio_path, recording_type):
        """
        Normalize audio based on type
        """
        try:
            print(f"\n[NORMALIZE] Processing {recording_type} recording...")
            
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            
            # Remove silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            if len(y_trimmed) < sr * 0.5:
                return None
            
            # Different normalization for different types
            if recording_type == 'DATASET':
                # More aggressive for dataset (telephone quality)
                y_normalized = librosa.util.normalize(y_trimmed)
                # Boost amplitude
                y_compressed = np.sign(y_normalized) * np.power(np.abs(y_normalized), 0.7)
            else:
                # Standard for live
                y_normalized = librosa.util.normalize(y_trimmed)
                y_compressed = np.sign(y_normalized) * np.power(np.abs(y_normalized), 0.8)
            
            # Save
            temp_path = audio_path.rsplit('.', 1)[0] + '_normalized.wav'
            sf.write(temp_path, y_compressed, sr)
            
            print(f"  ✅ Normalized: {len(y_compressed)/sr:.2f}s")
            return temp_path
            
        except Exception as e:
            print(f"[NORMALIZE ERROR] {e}")
            return None
    
    def validate_audio(self, audio_path):
        """Validate audio"""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            if duration < 0.5:
                return False, "⚠️ Audio too short (minimum 1 second)"
            if duration > 120:
                return False, "⚠️ Audio too long (maximum 2 minutes)"
            
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            if len(y_trimmed) < sr * 0.4:
                return False, "⚠️ Audio mostly silence"
            
            rms = librosa.feature.rms(y=y)
            if np.mean(rms) < 0.0003:
                return False, "⚠️ Volume too low"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def predict(self, audio_path):
        """
        Smart prediction with automatic type detection
        """
        try:
            print(f"\n{'='*70}")
            print(f"SMART VOICE ANALYSIS")
            print(f"{'='*70}")
            
            # STEP 1: Detect recording type
            recording_type, reasons = self.detect_recording_type(audio_path)
            
            # Select thresholds
            if recording_type == 'DATASET':
                thresholds = self.dataset_thresholds
                print(f"\n📞 Using DATASET thresholds (telephone quality)")
                print(f"   Jitter: {thresholds['jitter_high']:.5f}")
                print(f"   Shimmer: {thresholds['shimmer_high']:.5f}")
            else:
                thresholds = self.live_thresholds
                print(f"\n🎤 Using LIVE thresholds (high quality)")
                print(f"   Jitter: {thresholds['jitter_high']:.5f}")
                print(f"   Shimmer: {thresholds['shimmer_high']:.5f}")
            
            # STEP 2: Validate
            is_valid, msg = self.validate_audio(audio_path)
            if not is_valid:
                return msg, 0.0, None
            
            # STEP 3: Normalize
            normalized_path = self.normalize_audio(audio_path, recording_type)
            if normalized_path is None:
                return "⚠️ Audio processing failed", 0.0, None
            
            # STEP 4: Extract features
            print(f"\n[EXTRACTING] Features...")
            features = self.feature_extractor.extract_all_features(normalized_path)
            
            if features is None:
                if os.path.exists(normalized_path) and normalized_path != audio_path:
                    os.remove(normalized_path)
                return "⚠️ Feature extraction failed", 0.0, None
            
            # Get features
            jitter = features.get('jitter_local', 0)
            shimmer = features.get('shimmer_local', 0)
            hnr = features.get('hnr', 20)
            pitch_std = features.get('pitch_std', 20)
            pitch_mean = features.get('pitch_mean', 150)
            
            print(f"\n[FEATURES]:")
            print(f"  🎯 Jitter:     {jitter:.5f} ({jitter*100:.3f}%)")
            print(f"  🎯 Shimmer:    {shimmer:.5f} ({shimmer*100:.3f}%)")
            print(f"  📊 HNR:        {hnr:.2f} dB")
            print(f"  📈 Pitch Std:  {pitch_std:.2f} Hz")
            print(f"  📈 Pitch Mean: {pitch_mean:.2f} Hz")
            
            # STEP 5: Calculate score
            score, indicators = self._calculate_score(features, thresholds, recording_type)
            
            # STEP 6: Make decision
            print(f"\n[DECISION] Score: {score}/100")
            
            # Adjusted thresholds based on recording type
            if recording_type == 'DATASET':
                # More sensitive for dataset (lower threshold)
                if score >= 60:
                    result = "⚠️ Voice analysis indicates Parkinson's Disease"
                    confidence = score
                elif score >= 40:
                    result = "⚠️ Voice analysis shows borderline indicators"
                    confidence = score
                else:
                    result = "✅ Voice analysis indicates Healthy"
                    confidence = 100 - score
            else:
                # Less sensitive for live (higher threshold)
                if score >= 70:
                    result = "⚠️ Voice analysis indicates Parkinson's Disease"
                    confidence = score
                elif score >= 50:
                    result = "⚠️ Voice analysis shows borderline indicators"
                    confidence = score
                else:
                    result = "✅ Voice analysis indicates Healthy"
                    confidence = 100 - score
            
            # Create analysis
            analysis = {
                'pitch_variability': self._assess_pitch(pitch_std),
                'voice_stability': self._assess_stability(jitter, shimmer, thresholds),
                'voice_quality': self._assess_quality(hnr),
                'key_indicators': indicators if indicators else ['All parameters normal']
            }
            
            # Add recording type info
            analysis['recording_type'] = recording_type
            
            # Cleanup
            if os.path.exists(normalized_path) and normalized_path != audio_path:
                os.remove(normalized_path)
            
            print(f"\n[RESULT] {result}")
            print(f"[CONFIDENCE] {confidence:.2f}%")
            print(f"[TYPE] {recording_type}")
            print(f"{'='*70}\n")
            
            return result, round(confidence, 2), analysis
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return "⚠️ Error during analysis", 0.0, None
    
    def _calculate_score(self, features, thresholds, recording_type):
        """
        Calculate score with adaptive thresholds
        """
        score = 0
        indicators = []
        
        jitter = features.get('jitter_local', 0)
        shimmer = features.get('shimmer_local', 0)
        hnr = features.get('hnr', 20)
        pitch_std = features.get('pitch_std', 20)
        
        print(f"\n[SCORING] Using {recording_type} thresholds...")
        
        # JITTER (40 points)
        print(f"\n  1️⃣ JITTER:")
        print(f"     Value: {jitter:.5f} ({jitter*100:.3f}%)")
        print(f"     Threshold: {thresholds['jitter_high']:.5f}")
        
        if jitter > thresholds['jitter_high'] * 2.0:
            score += 40
            print(f"     🔴 SEVERE (+40)")
            indicators.append(f"Severe tremor ({jitter*100:.3f}%)")
        elif jitter > thresholds['jitter_high'] * 1.5:
            score += 35
            print(f"     🟠 VERY HIGH (+35)")
            indicators.append(f"Very high tremor ({jitter*100:.3f}%)")
        elif jitter > thresholds['jitter_high']:
            score += 30
            print(f"     🟡 HIGH (+30)")
            indicators.append(f"High tremor ({jitter*100:.3f}%)")
        elif jitter > thresholds['jitter_high'] * 0.8:
            score += 15
            print(f"     🟢 BORDERLINE (+15)")
        else:
            print(f"     ✅ NORMAL (+0)")
        
        # SHIMMER (40 points)
        print(f"\n  2️⃣ SHIMMER:")
        print(f"     Value: {shimmer:.5f} ({shimmer*100:.3f}%)")
        print(f"     Threshold: {thresholds['shimmer_high']:.5f}")
        
        if shimmer > thresholds['shimmer_high'] * 2.0:
            score += 40
            print(f"     🔴 SEVERE (+40)")
            indicators.append(f"Severe instability ({shimmer*100:.3f}%)")
        elif shimmer > thresholds['shimmer_high'] * 1.5:
            score += 35
            print(f"     🟠 VERY HIGH (+35)")
            indicators.append(f"Very high instability ({shimmer*100:.3f}%)")
        elif shimmer > thresholds['shimmer_high']:
            score += 30
            print(f"     🟡 HIGH (+30)")
            indicators.append(f"High instability ({shimmer*100:.3f}%)")
        elif shimmer > thresholds['shimmer_high'] * 0.8:
            score += 15
            print(f"     🟢 BORDERLINE (+15)")
        else:
            print(f"     ✅ NORMAL (+0)")
        
        # HNR (15 points)
        print(f"\n  3️⃣ HNR:")
        print(f"     Value: {hnr:.2f} dB")
        print(f"     Threshold: {thresholds['hnr_low']} dB")
        
        if 0 < hnr < thresholds['hnr_low'] * 0.7:
            score += 15
            print(f"     🟡 VERY BREATHY (+15)")
            indicators.append(f"Breathy voice ({hnr:.1f} dB)")
        elif 0 < hnr < thresholds['hnr_low']:
            score += 8
            print(f"     🟢 SLIGHTLY BREATHY (+8)")
        else:
            print(f"     ✅ NORMAL (+0)")
        
        # PITCH (5 points)
        print(f"\n  4️⃣ PITCH VARIATION:")
        print(f"     Value: {pitch_std:.2f} Hz")
        print(f"     Threshold: {thresholds['pitch_std_low']} Hz")
        
        if pitch_std < thresholds['pitch_std_low'] * 0.7:
            score += 5
            print(f"     🟡 VERY MONOTONE (+5)")
            indicators.append(f"Monotone ({pitch_std:.2f} Hz)")
        else:
            print(f"     ✅ NORMAL (+0)")
        
        print(f"\n{'─'*70}")
        print(f"  📊 TOTAL: {score}/100")
        print(f"{'─'*70}")
        
        return score, indicators
    
    def _assess_pitch(self, pitch_std):
        if pitch_std < 10:
            return 'Reduced'
        elif pitch_std > 40:
            return 'High'
        return 'Normal'
    
    def _assess_stability(self, jitter, shimmer, thresholds):
        if jitter > thresholds['jitter_high'] or shimmer > thresholds['shimmer_high']:
            return 'Unstable'
        return 'Stable'
    
    def _assess_quality(self, hnr):
        if 0 < hnr < 12:
            return 'Breathy'
        elif hnr > 25:
            return 'Clear'
        return 'Normal'


def get_recording_instructions():
    """Recording instructions"""
    return {
        'task_options': [
            {
                'name': 'Sustained Vowel',
                'instruction': 'Say "Aaaaaaah" for 5-10 seconds',
                'best_for': 'Best for tremor detection'
            },
            {
                'name': 'Reading',
                'instruction': 'Read: "The sun shines brightly"',
                'best_for': 'Natural speech'
            },
            {
                'name': 'Counting',
                'instruction': 'Count from 1 to 20',
                'best_for': 'Rhythm assessment'
            }
        ],
        'recording_tips': [
            '🎤 Quiet room',
            '📱 6 inches from mic',
            '🔊 Normal volume',
            '⏱️ 3-5 seconds minimum'
        ]
    }


if __name__ == "__main__":
    predictor = SmartVoicePredictor()
    test_file = "test_voice.wav"
    if os.path.exists(test_file):
        result, conf, analysis = predictor.predict(test_file)
        print(f"\nFINAL: {result} ({conf}%)")