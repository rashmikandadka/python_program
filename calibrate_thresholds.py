"""
Voice Threshold Calibration Tool
Analyzes your dataset to find optimal thresholds
"""

import numpy as np
import os
from audio_features import VoiceFeatureExtractor
import matplotlib.pyplot as plt


def analyze_dataset(dataset_folder):
    """
    Analyze entire dataset to find feature distributions
    """
    print("="*70)
    print("DATASET ANALYSIS FOR THRESHOLD CALIBRATION")
    print("="*70)
    
    extractor = VoiceFeatureExtractor()
    
    # Storage for features
    healthy_features = {
        'jitter': [],
        'shimmer': [],
        'hnr': [],
        'pitch_std': []
    }
    
    parkinson_features = {
        'jitter': [],
        'shimmer': [],
        'hnr': [],
        'pitch_std': []
    }
    
    # Process healthy samples
    print("\n📊 Analyzing HEALTHY samples...")
    healthy_folder = os.path.join(dataset_folder, 'healthy')
    if os.path.exists(healthy_folder):
        healthy_files = [f for f in os.listdir(healthy_folder) if f.endswith('.wav')]
        print(f"   Found {len(healthy_files)} files")
        
        for i, audio_file in enumerate(healthy_files, 1):
            audio_path = os.path.join(healthy_folder, audio_file)
            print(f"   [{i}/{len(healthy_files)}] {audio_file}")
            
            features = extractor.extract_all_features(audio_path)
            if features:
                healthy_features['jitter'].append(features.get('jitter_local', 0))
                healthy_features['shimmer'].append(features.get('shimmer_local', 0))
                healthy_features['hnr'].append(features.get('hnr', 0))
                healthy_features['pitch_std'].append(features.get('pitch_std', 0))
    
    # Process Parkinson samples
    print("\n📊 Analyzing PARKINSON samples...")
    parkinson_folder = os.path.join(dataset_folder, 'parkinson')
    if os.path.exists(parkinson_folder):
        parkinson_files = [f for f in os.listdir(parkinson_folder) if f.endswith('.wav')]
        print(f"   Found {len(parkinson_files)} files")
        
        for i, audio_file in enumerate(parkinson_files, 1):
            audio_path = os.path.join(parkinson_folder, audio_file)
            print(f"   [{i}/{len(parkinson_files)}] {audio_file}")
            
            features = extractor.extract_all_features(audio_path)
            if features:
                parkinson_features['jitter'].append(features.get('jitter_local', 0))
                parkinson_features['shimmer'].append(features.get('shimmer_local', 0))
                parkinson_features['hnr'].append(features.get('hnr', 0))
                parkinson_features['pitch_std'].append(features.get('pitch_std', 0))
    
    return healthy_features, parkinson_features


def calculate_statistics(healthy_features, parkinson_features):
    """
    Calculate statistics for each feature
    """
    print("\n" + "="*70)
    print("FEATURE STATISTICS")
    print("="*70)
    
    features_to_analyze = ['jitter', 'shimmer', 'hnr', 'pitch_std']
    recommended_thresholds = {}
    
    for feature in features_to_analyze:
        healthy = np.array(healthy_features[feature])
        parkinson = np.array(parkinson_features[feature])
        
        # Remove zeros
        healthy = healthy[healthy > 0]
        parkinson = parkinson[parkinson > 0]
        
        if len(healthy) == 0 or len(parkinson) == 0:
            print(f"\n⚠️ {feature.upper()}: Insufficient data")
            continue
        
        print(f"\n📊 {feature.upper()}:")
        print(f"   Healthy:")
        print(f"      Mean:   {np.mean(healthy):.5f}")
        print(f"      Median: {np.median(healthy):.5f}")
        print(f"      Std:    {np.std(healthy):.5f}")
        print(f"      Min:    {np.min(healthy):.5f}")
        print(f"      Max:    {np.max(healthy):.5f}")
        print(f"      95th:   {np.percentile(healthy, 95):.5f}")
        
        print(f"   Parkinson:")
        print(f"      Mean:   {np.mean(parkinson):.5f}")
        print(f"      Median: {np.median(parkinson):.5f}")
        print(f"      Std:    {np.std(parkinson):.5f}")
        print(f"      Min:    {np.min(parkinson):.5f}")
        print(f"      Max:    {np.max(parkinson):.5f}")
        print(f"      5th:    {np.percentile(parkinson, 5):.5f}")
        
        # Calculate optimal threshold
        if feature in ['jitter', 'shimmer']:
            # Higher values indicate Parkinson's
            # Use midpoint between healthy 95th percentile and Parkinson 25th percentile
            healthy_95 = np.percentile(healthy, 95)
            parkinson_25 = np.percentile(parkinson, 25)
            threshold = (healthy_95 + parkinson_25) / 2
            
            print(f"\n   💡 RECOMMENDED THRESHOLD: {threshold:.5f}")
            print(f"      (Between healthy 95th and Parkinson 25th)")
            recommended_thresholds[feature] = threshold
            
        elif feature == 'hnr':
            # Lower values indicate Parkinson's
            healthy_5 = np.percentile(healthy, 5)
            parkinson_75 = np.percentile(parkinson, 75)
            threshold = (healthy_5 + parkinson_75) / 2
            
            print(f"\n   💡 RECOMMENDED THRESHOLD: {threshold:.2f} dB")
            print(f"      (Between healthy 5th and Parkinson 75th)")
            recommended_thresholds[feature] = threshold
            
        elif feature == 'pitch_std':
            # Lower values indicate Parkinson's (monotone)
            healthy_10 = np.percentile(healthy, 10)
            parkinson_75 = np.percentile(parkinson, 75)
            threshold = (healthy_10 + parkinson_75) / 2
            
            print(f"\n   💡 RECOMMENDED THRESHOLD: {threshold:.2f} Hz")
            print(f"      (Between healthy 10th and Parkinson 75th)")
            recommended_thresholds[feature] = threshold
    
    return recommended_thresholds


def generate_threshold_code(thresholds):
    """
    Generate Python code with recommended thresholds
    """
    print("\n" + "="*70)
    print("RECOMMENDED THRESHOLD CODE")
    print("="*70)
    
    print("""
# Copy these thresholds into your CalibratedVoicePredictor class:

self.thresholds = {
    # PRIMARY INDICATORS""")
    
    if 'jitter' in thresholds:
        print(f"    'jitter_high': {thresholds['jitter']:.5f},  # Jitter threshold")
    
    if 'shimmer' in thresholds:
        print(f"    'shimmer_high': {thresholds['shimmer']:.5f},  # Shimmer threshold")
    
    print("\n    # SECONDARY INDICATORS")
    
    if 'hnr' in thresholds:
        print(f"    'hnr_low': {thresholds['hnr']:.2f},  # HNR threshold (dB)")
    
    if 'pitch_std' in thresholds:
        print(f"    'pitch_std_low': {thresholds['pitch_std']:.2f},  # Pitch variation threshold (Hz)")
    
    print("""    
    # PITCH RANGE
    'pitch_mean_low': 85,
    'pitch_mean_high': 280
}
""")


def test_thresholds(healthy_features, parkinson_features, thresholds):
    """
    Test how well the thresholds separate classes
    """
    print("\n" + "="*70)
    print("THRESHOLD PERFORMANCE TEST")
    print("="*70)
    
    # Test Jitter
    if 'jitter' in thresholds:
        jitter_threshold = thresholds['jitter']
        
        healthy_jitter = np.array(healthy_features['jitter'])
        parkinson_jitter = np.array(parkinson_features['jitter'])
        
        healthy_jitter = healthy_jitter[healthy_jitter > 0]
        parkinson_jitter = parkinson_jitter[parkinson_jitter > 0]
        
        healthy_correct = np.sum(healthy_jitter < jitter_threshold)
        parkinson_correct = np.sum(parkinson_jitter > jitter_threshold)
        
        print(f"\n📊 JITTER (threshold: {jitter_threshold:.5f}):")
        print(f"   Healthy correctly classified:   {healthy_correct}/{len(healthy_jitter)} ({healthy_correct/len(healthy_jitter)*100:.1f}%)")
        print(f"   Parkinson correctly classified: {parkinson_correct}/{len(parkinson_jitter)} ({parkinson_correct/len(parkinson_jitter)*100:.1f}%)")
        print(f"   Overall accuracy: {(healthy_correct + parkinson_correct)/(len(healthy_jitter) + len(parkinson_jitter))*100:.1f}%")
    
    # Test Shimmer
    if 'shimmer' in thresholds:
        shimmer_threshold = thresholds['shimmer']
        
        healthy_shimmer = np.array(healthy_features['shimmer'])
        parkinson_shimmer = np.array(parkinson_features['shimmer'])
        
        healthy_shimmer = healthy_shimmer[healthy_shimmer > 0]
        parkinson_shimmer = parkinson_shimmer[parkinson_shimmer > 0]
        
        healthy_correct = np.sum(healthy_shimmer < shimmer_threshold)
        parkinson_correct = np.sum(parkinson_shimmer > shimmer_threshold)
        
        print(f"\n📊 SHIMMER (threshold: {shimmer_threshold:.5f}):")
        print(f"   Healthy correctly classified:   {healthy_correct}/{len(healthy_shimmer)} ({healthy_correct/len(healthy_shimmer)*100:.1f}%)")
        print(f"   Parkinson correctly classified: {parkinson_correct}/{len(parkinson_shimmer)} ({parkinson_correct/len(parkinson_shimmer)*100:.1f}%)")
        print(f"   Overall accuracy: {(healthy_correct + parkinson_correct)/(len(healthy_shimmer) + len(parkinson_shimmer))*100:.1f}%")


def main():
    """
    Main calibration function
    """
    print("\n" + "="*70)
    print("VOICE THRESHOLD CALIBRATION TOOL")
    print("="*70)
    
    dataset_folder = 'voice_dataset'
    
    if not os.path.exists(dataset_folder):
        print(f"\n❌ Dataset folder not found: {dataset_folder}")
        print("Please create the following structure:")
        print("   voice_dataset/")
        print("   ├── healthy/")
        print("   │   └── *.wav")
        print("   └── parkinson/")
        print("       └── *.wav")
        return
    
    # Analyze dataset
    healthy_features, parkinson_features = analyze_dataset(dataset_folder)
    
    # Calculate statistics
    thresholds = calculate_statistics(healthy_features, parkinson_features)
    
    # Test thresholds
    if thresholds:
        test_thresholds(healthy_features, parkinson_features, thresholds)
        
        # Generate code
        generate_threshold_code(thresholds)
    
    print("\n" + "="*70)
    print("✅ CALIBRATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Copy the recommended thresholds into calibrated_voice_predictor.py")
    print("2. Test with sample recordings")
    print("3. Adjust if needed based on real-world performance")


if __name__ == "__main__":
    main()