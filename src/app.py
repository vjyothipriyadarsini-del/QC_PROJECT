from flask import Flask, send_from_directory
import os

# Define paths relative to this file (src/app.py)
# We want to serve index.html (in ../) and assets (in ../assets)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

app = Flask(__name__, static_folder=ASSETS_DIR)

@app.route('/')
def home():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/style.css')
def css():
    return send_from_directory(BASE_DIR, 'style.css')

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory(ASSETS_DIR, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
