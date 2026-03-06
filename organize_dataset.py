"""
Automatic Dataset Organizer (Updated)
Organizes Parkinson’s voice dataset from your existing structure.
"""

import os
import shutil
from pathlib import Path

def create_folder_structure():
    """Create target folders for organized dataset"""
    folders = [
        'voice_dataset',
        'voice_dataset/healthy',
        'voice_dataset/parkinson',
        'models',
        'models/voice_model'
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✓ Created: {folder}")


def copy_from_existing():
    """
    Copy files from your existing dataset/voice folders
    """
    src_base = Path("dataset/voice")  # source path
    dest_base = Path("voice_dataset")

    healthy_src = src_base / "healthy"
    parkinson_src = src_base / "parkinson"

    healthy_dest = dest_base / "healthy"
    parkinson_dest = dest_base / "parkinson"

    copied = {"healthy": 0, "parkinson": 0}

    # Copy healthy files
    if healthy_src.exists():
        for file in healthy_src.glob("*.wav"):
            shutil.copy2(file, healthy_dest / file.name)
            copied["healthy"] += 1
        print(f"✓ Copied {copied['healthy']} healthy files")

    # Copy parkinson files
    if parkinson_src.exists():
        for file in parkinson_src.glob("*.wav"):
            shutil.copy2(file, parkinson_dest / file.name)
            copied["parkinson"] += 1
        print(f"✓ Copied {copied['parkinson']} parkinson files")

    return copied


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("  🎤 PARKINSON'S VOICE DATASET ORGANIZER (Updated Version)")
    print("="*70)

    create_folder_structure()

    # Check if dataset/voice exists
    dataset_path = Path("dataset/voice")
    if not dataset_path.exists():
        print("\n❌ No 'dataset/voice' folder found.")
        print("Please make sure your folder structure looks like this:")
        print("   dataset/voice/healthy/")
        print("   dataset/voice/parkinson/")
        return

    print("\n📂 Copying from existing dataset folders...")
    copied = copy_from_existing()

    print("\n" + "="*70)
    print("  📊 SUMMARY")
    print("="*70)
    print(f"✓ Healthy files copied   : {copied['healthy']}")
    print(f"✓ Parkinson files copied : {copied['parkinson']}")
    total = copied['healthy'] + copied['parkinson']
    print(f"✓ Total files            : {total}")

    if total > 0:
        print("\n✅ Dataset organized successfully!")
        print("🚀 Next: Run training using:")
        print("   python train_voice_simple.py")
    else:
        print("\n⚠️ No .wav files found in your dataset/voice folders.")
        print("Please verify your dataset paths and filenames.")


if __name__ == "__main__":
    main()
