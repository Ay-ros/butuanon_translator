import sys
import os
sys.path.insert(0, 'src')
from butuanon_nlp.models import create_default_model_registry
import soundfile as sf
import numpy as np
import whisper

reg = create_default_model_registry()
w = reg.get('whisper')
w.use_huggingface = True
# lazy load
if not hasattr(w, '_whisper_model') or w._whisper_model is None:
    w._whisper_model = whisper.load_model('tiny', device='cpu')

path = r'C:\Users\YoYo\AppData\Local\Temp\butuanon_tts_apfa1zi1.wav'
print('path exists:', os.path.exists(path))
if not os.path.exists(path):
    raise SystemExit('WAV not found: ' + path)

audio, sr = sf.read(path)
print('sample rate:', sr, 'shape:', getattr(audio, 'shape', None), 'dtype:', getattr(audio, 'dtype', None))
if audio.ndim > 1:
    audio = np.mean(audio, axis=1)
# Whisper expects float32 audio
audio = audio.astype(np.float32)
print('Calling Whisper transcribe...')
res = w._whisper_model.transcribe(audio, language='en', fp16=False)
print('transcription ->', res.get('text'))
