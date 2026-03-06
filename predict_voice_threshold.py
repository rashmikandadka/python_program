"""
Threshold-based Voice Predictor for Parkinson's Detection
Uses clinical thresholds instead of ML model (more reliable with small datasets)
"""

import numpy as np
import os
from audio_features import VoiceFeatureExtractor


class ThresholdVoicePredictor:
    """
    Uses clinical thresholds for Parkinson's detection instead of ML model.
    Based on research findings about voice characteristics in Parkinson's.
    """
    
    def __init__(self):
        self.feature_extractor = VoiceFeatureExtractor()
        
        # Clinical thresholds based on research
        self.thresholds = {
            'jitter_high': 0.012,      # Jitter > 1.2% indicates tremor
            'shimmer_high': 0.06,      # Shimmer > 6% indicates instability
            'hnr_low': 12,             # HNR < 12 dB indicates breathiness
            'pitch_std_low': 10,       # Low pitch variation indicates monotone
            'pitch_mean_low': 100,     # Very low pitch
            'pitch_mean_high': 250     # Very high pitch
        }
        
        print("✅ Threshold-based voice predictor initialized")
        print(f"   Using clinical thresholds for detection")
    
    def validate_audio(self, audio_path):
        """Validate if the audio file is suitable for analysis"""
        try:
            import librosa
            
            print(f"[DEBUG] Validating audio: {audio_path}")
            
            # Try to load the audio
            y, sr = librosa.load(audio_path, sr=None)
            
            print(f"[DEBUG] Audio loaded - Duration: {len(y)/sr:.2f}s, Sample rate: {sr}Hz")
            
            # Check duration
            duration = len(y) / sr
            
            if duration < 0.5:
                return False, "⚠️ Audio too short. Please record at least 1-2 seconds"
            
            if duration > 120:
                return False, "⚠️ Audio too long. Please keep it under 2 minutes"
            
            # Check if audio is mostly silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            
            if len(y_trimmed) < sr * 0.3:
                return False, "⚠️ Audio is mostly silence. Please speak clearly and loudly"
            
            # Check volume
            rms = librosa.feature.rms(y=y)
            avg_rms = np.mean(rms)
            
            print(f"[DEBUG] Average RMS: {avg_rms}")
            
            if avg_rms < 0.0005:
                return False, "⚠️ Audio volume too low. Please speak louder"
            
            print("[DEBUG] Audio validation PASSED")
            return True, "Valid audio"
            
        except Exception as e:
            print(f"[DEBUG] Validation error: {str(e)}")
            return False, f"Error validating audio: {str(e)}"
    
    def predict(self, audio_path):
        """
        Predict Parkinson's using threshold-based approach.
        Returns: (result_text, confidence, feature_analysis)
        """
        try:
            print(f"\n{'='*60}")
            print(f"THRESHOLD-BASED VOICE ANALYSIS")
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
                return "⚠️ Unable to analyze audio. Please ensure clear speech", 0.0, None
            
            print(f"[SUCCESS] Extracted {len(features)} features")
            
            # Step 3: Analyze features using thresholds
            print("\n[STEP 2] Analyzing features with clinical thresholds...")
            analysis = self._analyze_features(features)
            
            # Step 4: Calculate Parkinson's risk score
            risk_score = self._calculate_risk_score(features)
            
            print(f"\n[STEP 3] Risk score calculated: {risk_score}/100")
            
            # Step 5: Make decision based on risk score
            if risk_score >= 60:
                result = "⚠️ Voice analysis indicates Parkinson's Disease"
                confidence = risk_score
            elif risk_score >= 40:
                result = "⚠️ Voice analysis shows some concerning features (borderline)"
                confidence = risk_score
            else:
                result = "✅ Voice analysis indicates Healthy"
                confidence = 100 - risk_score
            
            print(f"[RESULT] {result}")
            print(f"[CONFIDENCE] {confidence:.2f}%")
            
            print(f"\n{'='*60}")
            print(f"ANALYSIS COMPLETED")
            print(f"{'='*60}\n")
            
            return result, round(confidence, 2), analysis
            
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Analysis failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return "⚠️ Error during voice analysis. Please try again", 0.0, None
    
    def _calculate_risk_score(self, features):
        """
        Calculate Parkinson's risk score based on multiple features.
        Returns a score from 0-100 (higher = more likely Parkinson's)
        """
        risk_factors = 0
        total_factors = 0
        
        # 1. Jitter (voice tremor) - Weight: 25%
        jitter = features.get('jitter_local', 0)
        if jitter > self.thresholds['jitter_high']:
            risk_factors += 25
            print(f"  [+] High jitter detected: {jitter:.4f} (threshold: {self.thresholds['jitter_high']})")
        elif jitter > self.thresholds['jitter_high'] * 0.7:
            risk_factors += 15  # Borderline
            print(f"  [~] Borderline jitter: {jitter:.4f}")
        else:
            print(f"  [-] Normal jitter: {jitter:.4f}")
        total_factors += 25
        
        # 2. Shimmer (amplitude variation) - Weight: 25%
        shimmer = features.get('shimmer_local', 0)
        if shimmer > self.thresholds['shimmer_high']:
            risk_factors += 25
            print(f"  [+] High shimmer detected: {shimmer:.4f} (threshold: {self.thresholds['shimmer_high']})")
        elif shimmer > self.thresholds['shimmer_high'] * 0.7:
            risk_factors += 15  # Borderline
            print(f"  [~] Borderline shimmer: {shimmer:.4f}")
        else:
            print(f"  [-] Normal shimmer: {shimmer:.4f}")
        total_factors += 25
        
        # 3. HNR (voice quality) - Weight: 20%
        hnr = features.get('hnr', 20)
        if hnr < self.thresholds['hnr_low']:
            risk_factors += 20
            print(f"  [+] Low HNR (breathy voice): {hnr:.2f} dB (threshold: {self.thresholds['hnr_low']})")
        elif hnr < self.thresholds['hnr_low'] * 1.2:
            risk_factors += 10  # Borderline
            print(f"  [~] Borderline HNR: {hnr:.2f} dB")
        else:
            print(f"  [-] Normal HNR: {hnr:.2f} dB")
        total_factors += 20
        
        # 4. Pitch variability (monotone speech) - Weight: 15%
        pitch_std = features.get('pitch_std', 20)
        if pitch_std < self.thresholds['pitch_std_low']:
            risk_factors += 15
            print(f"  [+] Low pitch variation (monotone): {pitch_std:.2f} Hz")
        elif pitch_std < self.thresholds['pitch_std_low'] * 1.3:
            risk_factors += 8  # Borderline
            print(f"  [~] Borderline pitch variation: {pitch_std:.2f} Hz")
        else:
            print(f"  [-] Normal pitch variation: {pitch_std:.2f} Hz")
        total_factors += 15
        
        # 5. Pitch mean (very low or high pitch) - Weight: 15%
        pitch_mean = features.get('pitch_mean', 150)
        if pitch_mean > 0:  # Only if pitch was detected
            if pitch_mean < self.thresholds['pitch_mean_low'] or pitch_mean > self.thresholds['pitch_mean_high']:
                risk_factors += 10
                print(f"  [~] Unusual pitch mean: {pitch_mean:.2f} Hz")
            else:
                print(f"  [-] Normal pitch mean: {pitch_mean:.2f} Hz")
        total_factors += 15
        
        # Calculate percentage
        risk_score = (risk_factors / total_factors) * 100
        
        print(f"\n  Risk factors: {risk_factors}/{total_factors}")
        print(f"  Risk score: {risk_score:.1f}%")
        
        return risk_score
    
    def _analyze_features(self, features):
        """
        Provide detailed analysis of voice features.
        """
        analysis = {
            'pitch_variability': 'Normal',
            'voice_stability': 'Normal',
            'voice_quality': 'Normal',
            'key_indicators': []
        }
        
        # Analyze each feature
        jitter = features.get('jitter_local', 0)
        shimmer = features.get('shimmer_local', 0)
        hnr = features.get('hnr', 20)
        pitch_std = features.get('pitch_std', 20)
        pitch_mean = features.get('pitch_mean', 150)
        
        # Pitch variability
        if pitch_std < self.thresholds['pitch_std_low']:
            analysis['pitch_variability'] = 'Reduced'
            analysis['key_indicators'].append('Monotone speech (reduced pitch variation)')
        elif pitch_std > 50:
            analysis['pitch_variability'] = 'High'
            analysis['key_indicators'].append('Excessive pitch variation')
        
        # Voice stability (jitter + shimmer)
        if jitter > self.thresholds['jitter_high']:
            analysis['voice_stability'] = 'Unstable'
            analysis['key_indicators'].append(f'Voice tremor detected (jitter: {jitter:.3f})')
        
        if shimmer > self.thresholds['shimmer_high']:
            analysis['voice_stability'] = 'Unstable'
            analysis['key_indicators'].append(f'Amplitude instability (shimmer: {shimmer:.3f})')
        
        # Voice quality
        if hnr < self.thresholds['hnr_low']:
            analysis['voice_quality'] = 'Breathy'
            analysis['key_indicators'].append(f'Breathy voice quality (HNR: {hnr:.1f} dB)')
        
        # Pitch characteristics
        if pitch_mean > 0:
            if pitch_mean < self.thresholds['pitch_mean_low']:
                analysis['key_indicators'].append(f'Unusually low pitch ({pitch_mean:.1f} Hz)')
            elif pitch_mean > self.thresholds['pitch_mean_high']:
                analysis['key_indicators'].append(f'Unusually high pitch ({pitch_mean:.1f} Hz)')
        
        if not analysis['key_indicators']:
            analysis['key_indicators'].append('All voice parameters within normal clinical ranges')
        
        return analysis


def get_recording_instructions():
    """Returns instructions for recording voice samples"""
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
            '🎤 Record in a quiet room',
            '📱 Hold microphone 6 inches from mouth',
            '📊 Speak at normal volume',
            '⏱️ Record for at least 3-5 seconds',
            '🔇 Minimize background noise'
        ],
        'important_notes': [
            '⚠️ This is a screening tool, not a diagnostic test',
            '👨‍⚕️ Consult a neurologist for proper diagnosis',
            '📊 Results based on clinical voice thresholds'
        ]
    }


# Test the threshold predictor
if __name__ == "__main__":
    print("="*60)
    print("Threshold-Based Voice Parkinson's Detection")
    print("="*60)
    
    predictor = ThresholdVoicePredictor()
    
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