from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from butuanon_nlp.web_app import app


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
