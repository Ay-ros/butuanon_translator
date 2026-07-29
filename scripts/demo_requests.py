import sys
sys.path.insert(0, 'src')
from butuanon_nlp.web_app import create_app
import json

app = create_app()
client = app.test_client()

# Demo endpoint
resp = client.post('/api/demo', json={'text': 'Where are you going?', 'audio_path': ''})
print('DEMO status', resp.status_code)
print(resp.get_json())

# Translate endpoint
resp2 = client.post('/api/translate', json={'text': 'Good morning', 'backend': 'nllb'})
print('\nTRANSLATE status', resp2.status_code)
print(resp2.get_json())

# TTS endpoint (synthesize with vits adapter configured)
resp3 = client.post('/api/tts', json={'text': 'Maayong buntag', 'backend': 'vits'})
print('\nTTS status', resp3.status_code)
print(resp3.get_json())
