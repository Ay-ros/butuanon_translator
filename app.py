import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from butuanon_nlp.web_app import app

if __name__ == '__main__':
    # Hugging Face Spaces (Gradio) exposes port 7860 by default
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)
