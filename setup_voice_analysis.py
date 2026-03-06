"""
Quick Setup Script for Voice Analysis
Run this to set up your project structure and check dependencies
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    print("="*60)
    print("Checking Dependencies...")
    print("="*60)
    
    required = {
        'librosa': 'Audio processing',
        'parselmouth': 'Praat analysis (jitter, shimmer)',
        'sklearn': 'Machine learning',
        'joblib': 'Model persistence',
        'pandas': 'Data handling',
        'numpy': 'Numerical computing',
        'tensorflow': 'Deep learning',
        'flask': 'Web framework'
    }
    
    missing = []
    
    for package, description in required.items():
        try:
            __import__(package)
            print(f"✅ {package:20} - {description}")
        except ImportError:
            print(f"❌ {package:20} - {description} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def create_folder_structure():
    """Create necessary folders"""
    print("\n" + "="*60)
    print("Creating Folder Structure...")
    print("="*60)
    
    folders = [
        'models/voice_model',
        'static/uploads',
        'static/voice_uploads',
        'templates',
        'voice_dataset/healthy',
        'voice_dataset/parkinson'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created: {folder}")
    
    print("\n✅ Folder structure created!")


def download_sample_dataset():
    """Provide instructions for downloading dataset"""
    print("\n" + "="*60)
    print("Dataset Setup")
    print("="*60)
    
    print("""
📦 To train the voice model, you need audio data:

Option 1 - Download Existing Dataset:
  🔗 UCI Parkinson's Speech: https://archive.ics.uci.edu/ml/datasets/Parkinson+Speech+Dataset
  🔗 Kaggle Dataset: https://www.kaggle.com/datasets/dipayanbiswas/parkinsons-disease-speech-signal-features

Option 2 - Create Your Own:
  1. Record "Aaaah" sounds (5-10 seconds each)
  2. Save healthy recordings in: voice_dataset/healthy/
  3. Save parkinson recordings in: voice_dataset/parkinson/
  4. Minimum: 50 samples per class
  5. Format: WAV files (16kHz or 22kHz sample rate)

Recording Guidelines:
  • Quiet room
  • Clear pronunciation
  • Consistent distance from microphone
  • Natural voice (don't force)
    """)


def create_test_script():
    """Create a simple test script"""
    test_code = '''#!/usr/bin/env python
"""
Quick test script for voice analysis system
"""

def test_feature_extraction():
    """Test if feature extraction works"""
    print("Testing feature extraction...")
    try:
        from audio_features import VoiceFeatureExtractor
        extractor = VoiceFeatureExtractor()
        print("✅ Feature extractor loaded")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_files():
    """Check if model files exist"""
    import os
    print("\\nChecking model files...")
    
    model_path = 'models/voice_model'
    required_files = ['voice_model.h5', 'scaler.pkl', 'model_type.txt']
    
    if not os.path.exists(model_path):
        print("❌ Model folder not found")
        print("   Run training first: python train_voice_model.py")
        return False
    
    missing = []
    for file in required_files:
        filepath = os.path.join(model_path, file)
        if os.path.exists(filepath):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (missing)")
            missing.append(file)
    
    if missing:
        print("\\n⚠️  Train the model first: python train_voice_model.py")
        return False
    return True

def test_spiral_model():
    """Check spiral model"""
    import os
    print("\\nChecking spiral model...")
    
    if os.path.exists('models/parkinson_model.h5'):
        print("✅ Spiral model found")
        return True
    else:
        print("❌ Spiral model not found at models/parkinson_model.h5")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Voice Analysis System - Quick Test")
    print("="*60)
    
    results = []
    results.append(test_feature_extraction())
    results.append(test_spiral_model())
    results.append(test_model_files())
    
    print("\\n" + "="*60)
    if all(results):
        print("✅ All tests passed! System ready.")
        print("\\nRun: python app_combined.py")
    else:
        print("⚠️  Some tests failed. Check messages above.")
    print("="*60)
'''
    
    with open('test_system.py', 'w') as f:
        f.write(test_code)
    
    print("\n✅ Created test_system.py")
    print("   Run: python test_system.py")


def create_readme():
    """Create README file"""
    readme = """# Parkinson's Disease Detection - Voice Analysis Extension

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_voice.txt
```

### 2. Prepare Dataset
- Place healthy voice recordings in `voice_dataset/healthy/`
- Place parkinson voice recordings in `voice_dataset/parkinson/`
- Minimum 50 samples per class

### 3. Train Model
```bash
python train_voice_model.py
```

### 4. Run Application
```bash
python app_combined.py
```

Then open: http://localhost:5000

## Features

- 🌀 **Spiral Drawing Analysis** - Tests motor control
- 🎤 **Voice Analysis** - Detects speech abnormalities
- 🔬 **Combined Testing** - Multi-modal assessment

## File Structure

```
project/
├── app_combined.py          # Main application
├── audio_features.py        # Feature extraction
├── train_voice_model.py     # Model training
├── predict_voice.py         # Voice prediction
├── models/
│   ├── parkinson_model.h5   # Spiral model
│   └── voice_model/         # Voice model
└── voice_dataset/           # Training data
```

## Testing

```bash
python test_system.py
```

## Troubleshooting

See IMPLEMENTATION_GUIDE.md for detailed help.
"""
    
    with open('README_VOICE.md', 'w') as f:
        f.write(readme)
    
    print("✅ Created README_VOICE.md")


def main():
    """Main setup function"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🧠 Parkinson's Disease Detection - Voice Analysis Setup   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Create folders
    create_folder_structure()
    
    # Create helper files
    create_test_script()
    create_readme()
    
    # Dataset instructions
    download_sample_dataset()
    
    # Final instructions
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    
    if deps_ok:
        print("""
✅ Next Steps:

1. Add your voice dataset:
   - voice_dataset/healthy/*.wav
   - voice_dataset/parkinson/*.wav

2. Train the voice model:
   $ python train_voice_model.py

3. Test the system:
   $ python test_system.py

4. Run the application:
   $ python app_combined.py

📚 For detailed guide, see: IMPLEMENTATION_GUIDE.md
        """)
    else:
        print("""
⚠️  Please install missing dependencies first:
   $ pip install -r requirements_voice.txt

Then run this script again.
        """)
    
    print("="*60)


if __name__ == "__main__":
    main()