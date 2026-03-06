"""
CALIBRATED Voice Predictor for Parkinson's Detection
Fixed thresholds based on medical research and dataset analysis
"""

import numpy as np
import os
import librosa
import soundfile as sf
from audio_features import VoiceFeatureExtractor


class CalibratedVoicePredictor:
    """
    Properly calibrated voice predictor with research-based thresholds
    """
    
    def __init__(self):
        self.feature_extractor = VoiceFeatureExtractor()
        
        # CALIBRATED THRESHOLDS based on Parkinson's research
        # Source: Multiple studies on PD voice characteristics
        self.thresholds = {
            # PRIMARY INDICATORS (most reliable)
            'jitter_high': 0.010,      # 1.0% - Healthy typically < 0.6%
            'shimmer_high': 0.065,     # 6.5% - Healthy typically < 3.5%
            
            # SECONDARY INDICATORS (supporting evidence)
            'hnr_low': 14,             # 14 dB - Healthy typically > 20 dB
            'pitch_std_low': 12,       # 12 Hz - Low variation = monotone
            
            # PITCH RANGE (normal speaking range)
            'pitch_mean_low': 85,      # Below this = unusually low
            'pitch_mean_high': 280     # Above this = unusually high
        }
        
        print("✅ Calibrated Voice Predictor initialized")
        print("   Thresholds optimized for dataset + live recordings")
    
    def normalize_audio(self, audio_path):
        """
        Normalize audio with adaptive processing
        """
        try:
            print(f"\n[NORMALIZE] Processing audio...")
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            duration = len(y) / sr
            
            print(f"  📊 Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
            
            # Remove silence (adaptive threshold)
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            if len(y_trimmed) < sr * 0.5:
                print(f"  ❌ Too short after trimming")
                return None
            
            # Normalize amplitude
            y_normalized = librosa.util.normalize(y_trimmed)
            
            # Apply gentle dynamic range compression
            y_compressed = np.sign(y_normalized) * np.power(np.abs(y_normalized), 0.8)
            
            # Save to temp file
            temp_path = audio_path.rsplit('.', 1)[0] + '_normalized.wav'
            sf.write(temp_path, y_compressed, sr)
            
            print(f"  ✅ Normalized: {len(y_compressed)/sr:.2f}s")
            return temp_path
            
        except Exception as e:
            print(f"[ERROR] Normalization failed: {e}")
            return None
    
    def validate_audio(self, audio_path):
        """
        Validate audio quality
        """
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            
            # Check duration
            if duration < 0.5:
                return False, "⚠️ Audio too short (minimum 1 second)"
            if duration > 120:
                return False, "⚠️ Audio too long (maximum 2 minutes)"
            
            # Check if mostly silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            if len(y_trimmed) < sr * 0.4:
                return False, "⚠️ Audio is mostly silence"
            
            # Check volume
            rms = librosa.feature.rms(y=y)
            avg_rms = np.mean(rms)
            
            if avg_rms < 0.0003:
                return False, "⚠️ Audio volume too low"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def predict(self, audio_path):
        """
        Predict with calibrated thresholds
        """
        try:
            print(f"\n{'='*70}")
            print(f"CALIBRATED VOICE ANALYSIS")
            print(f"{'='*70}")
            
            # Validate
            is_valid, msg = self.validate_audio(audio_path)
            if not is_valid:
                return msg, 0.0, None
            
            # Normalize
            normalized_path = self.normalize_audio(audio_path)
            if normalized_path is None:
                return "⚠️ Audio processing failed", 0.0, None
            
            # Extract features
            print(f"\n[EXTRACTING] Voice features...")
            features = self.feature_extractor.extract_all_features(normalized_path)
            
            if features is None:
                if os.path.exists(normalized_path):
                    os.remove(normalized_path)
                return "⚠️ Feature extraction failed", 0.0, None
            
            # Get key features
            jitter = features.get('jitter_local', 0)
            shimmer = features.get('shimmer_local', 0)
            hnr = features.get('hnr', 20)
            pitch_std = features.get('pitch_std', 20)
            pitch_mean = features.get('pitch_mean', 150)
            
            print(f"\n[FEATURES] Extracted:")
            print(f"  🎯 Jitter:     {jitter:.5f} ({jitter*100:.3f}%)")
            print(f"  🎯 Shimmer:    {shimmer:.5f} ({shimmer*100:.3f}%)")
            print(f"  📊 HNR:        {hnr:.2f} dB")
            print(f"  📈 Pitch Std:  {pitch_std:.2f} Hz")
            print(f"  📈 Pitch Mean: {pitch_mean:.2f} Hz")
            
            # Calculate Parkinson's score
            score, indicators = self._calculate_score(features)
            
            # Make decision with STRICT thresholds
            print(f"\n[DECISION] Parkinson's Score: {score}/100")
            
            if score >= 70:  # Very strict - need strong evidence
                result = "⚠️ Voice analysis indicates Parkinson's Disease"
                confidence = score
            elif score >= 50:  # Borderline
                result = "⚠️ Voice analysis shows borderline indicators"
                confidence = score
            else:  # Healthy
                result = "✅ Voice analysis indicates Healthy"
                confidence = 100 - score
            
            # Create analysis
            analysis = {
                'pitch_variability': self._assess_pitch(pitch_std, pitch_mean),
                'voice_stability': self._assess_stability(jitter, shimmer),
                'voice_quality': self._assess_quality(hnr),
                'key_indicators': indicators if indicators else ['All parameters within normal ranges']
            }
            
            # Cleanup
            if os.path.exists(normalized_path) and normalized_path != audio_path:
                os.remove(normalized_path)
            
            print(f"\n[RESULT] {result}")
            print(f"[CONFIDENCE] {confidence:.2f}%")
            print(f"{'='*70}\n")
            
            return result, round(confidence, 2), analysis
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return "⚠️ Error during analysis", 0.0, None
    
    def _calculate_score(self, features):
        """
        Calculate Parkinson's score with STRICT, research-based criteria
        """
        score = 0
        indicators = []
        
        jitter = features.get('jitter_local', 0)
        shimmer = features.get('shimmer_local', 0)
        hnr = features.get('hnr', 20)
        pitch_std = features.get('pitch_std', 20)
        pitch_mean = features.get('pitch_mean', 150)
        
        print(f"\n[SCORING] Analyzing features...")
        
        # =====================================================
        # JITTER ANALYSIS (40 points) - Most important
        # =====================================================
        print(f"\n  1️⃣ JITTER (Voice Tremor):")
        print(f"     Value: {jitter:.5f} ({jitter*100:.3f}%)")
        print(f"     Threshold: {self.thresholds['jitter_high']:.5f} ({self.thresholds['jitter_high']*100:.3f}%)")
        
        if jitter > self.thresholds['jitter_high'] * 1.5:  # Very high
            score += 40
            print(f"     ⚠️ SEVERE tremor - Strong PD indicator")
            print(f"     +40 points")
            indicators.append(f"Severe voice tremor ({jitter*100:.3f}%)")
        elif jitter > self.thresholds['jitter_high']:  # High
            score += 30
            print(f"     ⚠️ HIGH tremor - Moderate PD indicator")
            print(f"     +30 points")
            indicators.append(f"Elevated voice tremor ({jitter*100:.3f}%)")
        elif jitter > self.thresholds['jitter_high'] * 0.8:  # Borderline
            score += 15
            print(f"     ~ Borderline tremor")
            print(f"     +15 points")
        else:  # Normal
            print(f"     ✅ NORMAL - Healthy range")
            print(f"     +0 points")
        
        # =====================================================
        # SHIMMER ANALYSIS (40 points) - Most important
        # =====================================================
        print(f"\n  2️⃣ SHIMMER (Amplitude Instability):")
        print(f"     Value: {shimmer:.5f} ({shimmer*100:.3f}%)")
        print(f"     Threshold: {self.thresholds['shimmer_high']:.5f} ({self.thresholds['shimmer_high']*100:.3f}%)")
        
        if shimmer > self.thresholds['shimmer_high'] * 1.3:  # Very high
            score += 40
            print(f"     ⚠️ SEVERE instability - Strong PD indicator")
            print(f"     +40 points")
            indicators.append(f"Severe amplitude instability ({shimmer*100:.3f}%)")
        elif shimmer > self.thresholds['shimmer_high']:  # High
            score += 30
            print(f"     ⚠️ HIGH instability - Moderate PD indicator")
            print(f"     +30 points")
            indicators.append(f"Elevated amplitude instability ({shimmer*100:.3f}%)")
        elif shimmer > self.thresholds['shimmer_high'] * 0.8:  # Borderline
            score += 15
            print(f"     ~ Borderline instability")
            print(f"     +15 points")
        else:  # Normal
            print(f"     ✅ NORMAL - Healthy range")
            print(f"     +0 points")
        
        # =====================================================
        # HNR ANALYSIS (15 points) - Supporting indicator
        # =====================================================
        print(f"\n  3️⃣ HNR (Voice Quality):")
        print(f"     Value: {hnr:.2f} dB")
        print(f"     Threshold: {self.thresholds['hnr_low']} dB")
        
        if 0 < hnr < self.thresholds['hnr_low'] * 0.7:  # Very low
            score += 15
            print(f"     ⚠️ Very breathy voice")
            print(f"     +15 points")
            indicators.append(f"Breathy voice quality ({hnr:.1f} dB)")
        elif 0 < hnr < self.thresholds['hnr_low']:  # Low
            score += 8
            print(f"     ~ Slightly breathy")
            print(f"     +8 points")
        else:  # Normal
            print(f"     ✅ NORMAL voice quality")
            print(f"     +0 points")
        
        # =====================================================
        # PITCH VARIATION ANALYSIS (5 points) - Least weight
        # =====================================================
        print(f"\n  4️⃣ PITCH VARIATION (Monotone Speech):")
        print(f"     Std: {pitch_std:.2f} Hz, Mean: {pitch_mean:.2f} Hz")
        print(f"     Threshold: {self.thresholds['pitch_std_low']} Hz")
        
        if pitch_std < self.thresholds['pitch_std_low'] * 0.7 and 85 < pitch_mean < 280:
            score += 5
            print(f"     ~ Very monotone")
            print(f"     +5 points")
            indicators.append(f"Monotone speech ({pitch_std:.2f} Hz)")
        else:
            print(f"     ✅ Normal variation")
            print(f"     +0 points")
        
        print(f"\n{'─'*70}")
        print(f"  📊 TOTAL SCORE: {score}/100")
        print(f"{'─'*70}")
        
        return score, indicators
    
    def _assess_pitch(self, pitch_std, pitch_mean):
        """Assess pitch variability"""
        if pitch_std < self.thresholds['pitch_std_low']:
            return 'Reduced'
        elif pitch_std > 40:
            return 'High'
        return 'Normal'
    
    def _assess_stability(self, jitter, shimmer):
        """Assess voice stability"""
        if jitter > self.thresholds['jitter_high'] or shimmer > self.thresholds['shimmer_high']:
            return 'Unstable'
        return 'Stable'
    
    def _assess_quality(self, hnr):
        """Assess voice quality"""
        if 0 < hnr < self.thresholds['hnr_low']:
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
                'best_for': 'Best for detecting tremor'
            },
            {
                'name': 'Reading',
                'instruction': 'Read: "The sun shines brightly"',
                'best_for': 'Natural speech analysis'
            },
            {
                'name': 'Counting',
                'instruction': 'Count from 1 to 20',
                'best_for': 'Rhythm assessment'
            }
        ],
        'recording_tips': [
            '🎤 Record in a quiet room',
            '📱 Hold mic 6 inches away',
            '🔊 Speak at normal volume',
            '⏱️ Record for 3-5 seconds minimum'
        ]
    }


# Quick test
if __name__ == "__main__":
    predictor = CalibratedVoicePredictor()
    
    test_file = "test_voice.wav"
    if os.path.exists(test_file):
        result, conf, analysis = predictor.predict(test_file)
        print(f"\nFINAL: {result} ({conf}%)")