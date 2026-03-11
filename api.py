"""
WSPP - Serveur API
Lance avec : python3 api.py
Accessible sur http://localhost:5001
"""
from flask import Flask, jsonify
from flask_cors import CORS
import json, os, threading, time
from model import predict_all

app  = Flask(__name__)
CORS(app)
FILE = 'predictions.json'

def refresh_loop():
    while True:
        time.sleep(24 * 60 * 60)
        print("🔄 Ré-entraînement quotidien...")
        results = predict_all()
        with open(FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("✅ Modèle mis à jour")

@app.route('/predictions')
def get_predictions():
    if os.path.exists(FILE):
        with open(FILE, encoding='utf-8') as f:
            return jsonify({'status': 'ok', 'data': json.load(f)})
    return jsonify({'status': 'error', 'message': 'Pas encore de prédictions'}), 404

@app.route('/health')
def health():
    return jsonify({'status': 'alive'})

if __name__ == '__main__':
    print("🚀 Démarrage WSPP ML Server...")
    results = predict_all()
    with open(FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    threading.Thread(target=refresh_loop, daemon=True).start()
    print("\n✅ API disponible sur http://localhost:5001/predictions")
    app.run(port=5001, debug=False)
