#!/bin/bash
# WSPP ML — Installation et lancement
# Usage : bash setup.sh

echo ""
echo "🤖 WSPP Machine Learning Setup"
echo "================================"
echo ""

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trouvé. Installe Python depuis python.org"
    exit 1
fi

echo "✅ Python : $(python3 --version)"

# Environnement virtuel
if [ ! -d "venv" ]; then
    echo "📦 Création environnement virtuel..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "✅ Environnement virtuel activé"

# Dépendances
echo "📦 Installation des librairies ML..."
pip install -r requirements.txt -q
echo "✅ Librairies installées"

echo ""
echo "================================"
echo "✅ Installation terminée !"
echo ""
echo "Pour lancer le serveur IA :"
echo "  source venv/bin/activate"
echo "  python3 api.py"
echo ""
echo "L'IA analysera 76 actifs (ETFs + CAC40 + Dow Jones)"
echo "et sera accessible sur http://localhost:5001/predictions"
echo "================================"
