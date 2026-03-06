from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import librosa
import soundfile as sf
from predict import predict_result

# Import voice predictor
try:
    from simple_voice_predictor import SimpleVoicePredictor, get_recording_instructions
    voice_predictor = SimpleVoicePredictor()
    VOICE_MODEL_AVAILABLE = True
    print("✅ Voice analysis loaded")
except Exception as e:
    print(f"❌ Voice error: {e}")
    VOICE_MODEL_AVAILABLE = False

app = Flask(__name__)

# Folders
UPLOAD_FOLDER = 'static/uploads'
VOICE_FOLDER = 'static/voice_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VOICE_FOLDER, exist_ok=True)

# Extensions
ALLOWED_IMAGE = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_AUDIO = {'wav', 'mp3', 'ogg', 'webm', 'm4a', 'mp4', 'mpeg', 'flac'}


def allowed_file(filename, file_type='image'):
    """Check file extension"""
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if file_type == 'image':
        return ext in ALLOWED_IMAGE
    else:
        return ext in ALLOWED_AUDIO


def simple_convert_to_wav(audio_path):
    """
    ULTRA-SIMPLE audio conversion with detailed logging
    """
    print(f"\n{'='*70}")
    print(f"[AUDIO CONVERSION START]")
    print(f"{'='*70}")
    
    try:
        print(f"Input file: {audio_path}")
        print(f"File exists: {os.path.exists(audio_path)}")
        print(f"File size: {os.path.getsize(audio_path):,} bytes")
        
        # If already WAV, try to use it
        if audio_path.lower().endswith('.wav'):
            print(f"File is already WAV, testing...")
            try:
                y, sr = librosa.load(audio_path, sr=22050, duration=0.5)
                print(f"  WAV test: {len(y)} samples loaded")
                if len(y) > 1000:
                    print(f"✅ Valid WAV - using directly")
                    print(f"{'='*70}\n")
                    return audio_path
                else:
                    print(f"  WAV too short, will convert")
            except Exception as e:
                print(f"  WAV test failed: {e}")
                print(f"  Will attempt conversion")
        
        # Load audio
        print(f"\nLoading audio with librosa...")
        try:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            print(f"✅ Loaded successfully!")
            print(f"   Samples: {len(y)}")
            print(f"   Sample rate: {sr} Hz")
            print(f"   Duration: {len(y)/sr:.3f} seconds")
            print(f"   Max amplitude: {np.max(np.abs(y)):.4f}")
        except Exception as e:
            print(f"❌ Loading failed: {e}")
            print(f"{'='*70}\n")
            return None
        
        # Check minimum length
        if len(y) < 1500:  # About 0.07 seconds - very lenient!
            print(f"❌ Audio too short: {len(y)} samples ({len(y)/sr:.3f}s)")
            print(f"   Minimum: 1500 samples (0.07s)")
            print(f"{'='*70}\n")
            return None
        
        # Check if silent
        max_amp = np.max(np.abs(y))
        if max_amp < 0.001:
            print(f"❌ Audio is silent (max amp: {max_amp:.6f})")
            print(f"{'='*70}\n")
            return None
        
        print(f"\n✅ Audio validation passed")
        
        # Gentle trim
        print(f"\nTrimming silence...")
        try:
            y_before = len(y)
            y_trimmed, _ = librosa.effects.trim(y, top_db=40)
            y_after = len(y_trimmed)
            print(f"   Before: {y_before} samples")
            print(f"   After:  {y_after} samples")
            print(f"   Kept:   {y_after/y_before*100:.1f}%")
            
            # Only use trimmed if we kept at least 20%
            if len(y_trimmed) >= len(y) * 0.2:
                y = y_trimmed
                print(f"   ✅ Using trimmed version")
            else:
                print(f"   ⚠️ Trim too aggressive, using original")
        except Exception as e:
            print(f"   ⚠️ Trim failed: {e}, using original")
        
        # Normalize
        print(f"\nNormalizing...")
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
            print(f"   ✅ Normalized to range [-1, 1]")
        else:
            print(f"   ⚠️ Cannot normalize, audio is silent")
        
        # Save
        wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
        print(f"\nSaving to: {wav_path}")
        
        try:
            sf.write(wav_path, y, sr)
            saved_size = os.path.getsize(wav_path)
            print(f"✅ Saved successfully!")
            print(f"   Output size: {saved_size:,} bytes")
            print(f"{'='*70}\n")
            return wav_path
        except Exception as e:
            print(f"❌ Save failed: {e}")
            print(f"{'='*70}\n")
            return None
        
    except Exception as e:
        print(f"❌ CONVERSION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*70}\n")
        return None


@app.route('/')
def index():
    """Home"""
    instructions = get_recording_instructions() if VOICE_MODEL_AVAILABLE else None
    return render_template('index_combined.html', 
                         voice_available=VOICE_MODEL_AVAILABLE,
                         recording_instructions=instructions,
                         spiral_msg=None,
                         voice_msg=None,
                         combined_msg=None)


@app.route('/predict_spiral', methods=['POST'])
def predict_spiral():
    """Spiral test"""
    if 'file' not in request.files:
        return render_template('index_combined.html', 
                             spiral_msg='No file selected',
                             voice_available=VOICE_MODEL_AVAILABLE,
                             recording_instructions=get_recording_instructions() if VOICE_MODEL_AVAILABLE else None)
    
    file = request.files['file']
    if not file or file.filename == '' or not allowed_file(file.filename, 'image'):
        return render_template('index_combined.html', 
                             spiral_msg='Invalid image file',
                             voice_available=VOICE_MODEL_AVAILABLE,
                             recording_instructions=get_recording_instructions() if VOICE_MODEL_AVAILABLE else None)
    
    # Save and predict
    filename = secure_filename(file.filename)
    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(upload_path)
    
    result, confidence, treatment = predict_result(upload_path)
    
    return render_template('result_combined.html',
                         test_type='Spiral Drawing',
                         result=result,
                         confidence=confidence,
                         filename=filename,
                         treatment_suggestions=treatment,
                         analysis=None,
                         voice_available=VOICE_MODEL_AVAILABLE)


@app.route('/predict_voice', methods=['POST'])
def predict_voice():
    """
    VOICE PREDICTION with MAXIMUM debugging
    """
    # CRITICAL DEBUG - ALWAYS PRINTS
    print("\n" + "="*70)
    print("🎯 VOICE ROUTE ACCESSED!!!")
    print("="*70)
    print(f"Method: {request.method}")
    print(f"Form keys: {list(request.form.keys())}")
    print(f"Files keys: {list(request.files.keys())}")
    print(f"Content-Type: {request.content_type}")
    print("="*70 + "\n")
    
    if not VOICE_MODEL_AVAILABLE:
        print(f"❌ Voice model not available")
        return render_template('index_combined.html',
                             voice_msg='Voice analysis not available',
                             voice_available=False,
                             recording_instructions=None,
                             spiral_msg=None,
                             combined_msg=None)
    
    print(f"✅ Voice model available")
    
    # Check file in request
    if 'voice_file' not in request.files:
        print(f"❌ No 'voice_file' in request.files")
        print(f"   Available keys: {list(request.files.keys())}")
        return render_template('index_combined.html',
                             voice_msg='No file selected. Please record or upload audio.',
                             voice_available=VOICE_MODEL_AVAILABLE,
                             recording_instructions=get_recording_instructions(),
                             spiral_msg=None,
                             combined_msg=None)
    
    print(f"✅ 'voice_file' found in request")
    
    file = request.files['voice_file']
    
    if not file or file.filename == '':
        print(f"❌ File is empty or has no filename")
        print(f"   File: {file}")
        print(f"   Filename: '{file.filename}'")
        return render_template('index_combined.html',
                             voice_msg='No file uploaded. Please record or upload audio.',
                             voice_available=VOICE_MODEL_AVAILABLE,
                             recording_instructions=get_recording_instructions(),
                             spiral_msg=None,
                             combined_msg=None)
    
    print(f"✅ File received: {file.filename}")
    
    # Check extension
    if not allowed_file(file.filename, 'audio'):
        print(f"❌ Invalid extension")
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'none'
        print(f"   Extension: {ext}")
        print(f"   Allowed: {ALLOWED_AUDIO}")
        return render_template('index_combined.html',
                             voice_msg=f'Invalid audio format: {ext}',
                             voice_available=VOICE_MODEL_AVAILABLE,
                             recording_instructions=get_recording_instructions(),
                             spiral_msg=None,
                             combined_msg=None)
    
    print(f"✅ Extension valid")
    
    # Check size
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    
    print(f"✅ File size: {size:,} bytes ({size/1024:.2f} KB)")
    
    if size < 100:
        print(f"❌ File too small")
        return render_template('index_combined.html',
                             voice_msg=f'File too small ({size} bytes). Record for 2+ seconds.',
                             voice_available=VOICE_MODEL_AVAILABLE,
                             recording_instructions=get_recording_instructions(),
                             spiral_msg=None,
                             combined_msg=None)
    
    # Save file
    filename = secure_filename(file.filename)
    upload_path = os.path.join(VOICE_FOLDER, filename)
    
    print(f"\n📁 Saving to: {upload_path}")
    
    try:
        file.save(upload_path)
        actual_size = os.path.getsize(upload_path)
        print(f"✅ File saved successfully")
        print(f"   Expected size: {size:,} bytes")
        print(f"   Actual size:   {actual_size:,} bytes")
        
        if actual_size == 0:
            print(f"❌ Saved file is empty!")
            os.remove(upload_path)
            return render_template('index_combined.html',
                                 voice_msg='File save failed (0 bytes)',
                                 voice_available=VOICE_MODEL_AVAILABLE,
                                 recording_instructions=get_recording_instructions(),
                                 spiral_msg=None,
                                 combined_msg=None)
        
    except Exception as e:
        print(f"❌ Save error: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index_combined.html',
                             voice_msg=f'Error saving file',
                             voice_available=VOICE_MODEL_AVAILABLE,
                             recording_instructions=get_recording_instructions(),
                             spiral_msg=None,
                             combined_msg=None)
    
    # Convert audio
    print(f"\n🔄 Starting audio conversion...")
    wav_path = simple_convert_to_wav(upload_path)
    
    if wav_path is None:
        print(f"\n❌ CONVERSION FAILED")
        if os.path.exists(upload_path):
            os.remove(upload_path)
        return render_template('index_combined.html',
                             voice_msg='⚠️ Could not process audio. Try: 1) Record 2+ seconds, 2) Speak clearly, 3) Upload different file',
                             voice_available=VOICE_MODEL_AVAILABLE,
                             recording_instructions=get_recording_instructions(),
                             spiral_msg=None,
                             combined_msg=None)
    
    print(f"\n✅ CONVERSION SUCCESSFUL")
    print(f"   WAV file: {wav_path}")
    
    # Analyze voice
    print(f"\n🔬 Starting voice analysis...")
    
    try:
        result, confidence, analysis = voice_predictor.predict(wav_path)
        print(f"\n✅ ANALYSIS SUCCESSFUL")
        print(f"   Result: {result}")
        print(f"   Confidence: {confidence}%")
    except Exception as e:
        print(f"\n❌ ANALYSIS ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup
        if os.path.exists(upload_path):
            os.remove(upload_path)
        if wav_path != upload_path and os.path.exists(wav_path):
            os.remove(wav_path)
        
        return render_template('index_combined.html',
                             voice_msg='⚠️ Analysis failed. Please try again.',
                             voice_available=VOICE_MODEL_AVAILABLE,
                             recording_instructions=get_recording_instructions(),
                             spiral_msg=None,
                             combined_msg=None)
    
    # Treatment suggestions
    treatment = None
    if "Parkinson" in result and "⚠️" in result and "borderline" not in result.lower():
        from predict import get_treatment_suggestions
        treatment = get_treatment_suggestions()
    
    # Cleanup temp file
    if wav_path != upload_path and os.path.exists(wav_path):
        try:
            os.remove(wav_path)
            print(f"🗑️ Cleaned up temporary file")
        except:
            pass
    
    print(f"\n✅ RETURNING RESULT PAGE")
    print(f"{'='*70}\n")
    
    return render_template('result_combined.html',
                         test_type='Voice Analysis',
                         result=result,
                         confidence=confidence,
                         filename=filename,
                         treatment_suggestions=treatment,
                         analysis=analysis,
                         voice_available=VOICE_MODEL_AVAILABLE)


@app.route('/predict_combined', methods=['POST'])
def predict_combined():
    """Combined test"""
    if not VOICE_MODEL_AVAILABLE:
        return render_template('index_combined.html',
                             combined_msg='Voice analysis not available',
                             voice_available=VOICE_MODEL_AVAILABLE,
                             recording_instructions=get_recording_instructions(),
                             spiral_msg=None,
                             voice_msg=None)
    
    spiral_file = request.files.get('spiral_file')
    voice_file = request.files.get('voice_file_combined')
    
    if not spiral_file or not voice_file:
        return render_template('index_combined.html',
                             combined_msg='Please upload both files',
                             voice_available=VOICE_MODEL_AVAILABLE,
                             recording_instructions=get_recording_instructions(),
                             spiral_msg=None,
                             voice_msg=None)
    
    # Save files
    spiral_filename = secure_filename(spiral_file.filename)
    spiral_path = os.path.join(UPLOAD_FOLDER, spiral_filename)
    spiral_file.save(spiral_path)
    
    voice_filename = secure_filename(voice_file.filename)
    voice_path = os.path.join(VOICE_FOLDER, voice_filename)
    voice_file.save(voice_path)
    
    # Convert voice
    wav_path = simple_convert_to_wav(voice_path)
    
    if wav_path is None:
        return render_template('index_combined.html',
                             combined_msg='Voice processing failed',
                             voice_available=VOICE_MODEL_AVAILABLE,
                             recording_instructions=get_recording_instructions(),
                             spiral_msg=None,
                             voice_msg=None)
    
    # Get predictions
    spiral_result, spiral_conf, spiral_treatment = predict_result(spiral_path)
    voice_result, voice_conf, voice_analysis = voice_predictor.predict(wav_path)
    
    # Combined
    spiral_pos = "Parkinson" in spiral_result and "⚠️" in spiral_result
    voice_pos = "Parkinson" in voice_result and "⚠️" in voice_result
    
    if spiral_pos and voice_pos:
        combined_result = "⚠️ BOTH indicate Parkinson's"
        combined_conf = (spiral_conf + voice_conf) / 2
        recommendation = "Strong indication - See neurologist"
    elif spiral_pos or voice_pos:
        combined_result = "⚠️ ONE test indicates concern"
        combined_conf = max(spiral_conf, voice_conf)
        recommendation = "Mixed results - Medical evaluation recommended"
    else:
        combined_result = "✅ BOTH indicate Healthy"
        combined_conf = (spiral_conf + voice_conf) / 2
        recommendation = "Both tests normal"
    
    # Treatment
    treatment = None
    if spiral_pos or voice_pos:
        from predict import get_treatment_suggestions
        treatment = get_treatment_suggestions()
    
    # Cleanup
    if wav_path != voice_path and os.path.exists(wav_path):
        try:
            os.remove(wav_path)
        except:
            pass
    
    return render_template('result_combined_both.html',
                         spiral_result=spiral_result,
                         spiral_confidence=spiral_conf,
                         voice_result=voice_result,
                         voice_confidence=voice_conf,
                         combined_result=combined_result,
                         combined_confidence=combined_conf,
                         recommendation=recommendation,
                         treatment_suggestions=treatment,
                         voice_analysis=voice_analysis,
                         spiral_filename=spiral_filename,
                         voice_filename=voice_filename)


if __name__ == '__main__':
    print("="*60)
    print("Parkinson's Detection - BULLETPROOF + FULL DEBUG")
    print("="*60)
    print("✅ Spiral: Available")
    print(f"{'✅' if VOICE_MODEL_AVAILABLE else '❌'} Voice: {'Available' if VOICE_MODEL_AVAILABLE else 'Not Available'}")
    print("="*60)
    print("\n📋 Debug mode: FULL LOGGING ENABLED")
    print("   Every step will be printed to console")
    print("="*60)
    
    app.run(debug=True, port=5000)