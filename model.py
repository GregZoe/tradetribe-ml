"""
WSPP - Modèle de prédiction IA v2
Couvre : ETFs + CAC 40 + Dow Jones
Prédit à 1j, 3j, 5j et 10j simultanément
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════
# UNIVERS D'INVESTISSEMENT
# ═══════════════════════════════════════════════

ETFS = {
    'ITA':  {'name': 'ETF Défense',          'market': 'etf', 'currency': 'USD'},
    'SPY':  {'name': 'ETF S&P 500',          'market': 'etf', 'currency': 'USD'},
    'SHY':  {'name': 'ETF Obligations 1-3Y', 'market': 'etf', 'currency': 'USD'},
    'VGK':  {'name': 'ETF Europe',           'market': 'etf', 'currency': 'USD'},
    'BOTZ': {'name': 'ETF IA & Robotique',   'market': 'etf', 'currency': 'USD'},
    'QQQ':  {'name': 'ETF Nasdaq 100',       'market': 'etf', 'currency': 'USD'},
    'ICLN': {'name': 'ETF Énergie Propre',   'market': 'etf', 'currency': 'USD'},
    'VWO':  {'name': 'ETF Émergents',        'market': 'etf', 'currency': 'USD'},
    'BND':  {'name': 'ETF Obligations',      'market': 'etf', 'currency': 'USD'},
    'VYM':  {'name': 'ETF Dividendes',       'market': 'etf', 'currency': 'USD'},
    'VHT':  {'name': 'ETF Santé',            'market': 'etf', 'currency': 'USD'},
}

CAC40 = {
    'AI.PA':    {'name': 'Air Liquide',        'market': 'cac40', 'currency': 'EUR'},
    'AIR.PA':   {'name': 'Airbus',             'market': 'cac40', 'currency': 'EUR'},
    'ALO.PA':   {'name': 'Alstom',             'market': 'cac40', 'currency': 'EUR'},
    'BN.PA':    {'name': 'Danone',             'market': 'cac40', 'currency': 'EUR'},
    'BNP.PA':   {'name': 'BNP Paribas',        'market': 'cac40', 'currency': 'EUR'},
    'CA.PA':    {'name': 'Carrefour',          'market': 'cac40', 'currency': 'EUR'},
    'CAP.PA':   {'name': 'Capgemini',          'market': 'cac40', 'currency': 'EUR'},
    'CS.PA':    {'name': 'AXA',                'market': 'cac40', 'currency': 'EUR'},
    'DG.PA':    {'name': 'Vinci',              'market': 'cac40', 'currency': 'EUR'},
    'DSY.PA':   {'name': 'Dassault Systèmes',  'market': 'cac40', 'currency': 'EUR'},
    'ENGI.PA':  {'name': 'Engie',              'market': 'cac40', 'currency': 'EUR'},
    'EL.PA':    {'name': 'EssilorLuxottica',   'market': 'cac40', 'currency': 'EUR'},
    'GLE.PA':   {'name': 'Société Générale',   'market': 'cac40', 'currency': 'EUR'},
    'HO.PA':    {'name': 'Thales',             'market': 'cac40', 'currency': 'EUR'},
    'KER.PA':   {'name': 'Kering',             'market': 'cac40', 'currency': 'EUR'},
    'LR.PA':    {'name': 'Legrand',            'market': 'cac40', 'currency': 'EUR'},
    'MC.PA':    {'name': 'LVMH',               'market': 'cac40', 'currency': 'EUR'},
    'ML.PA':    {'name': 'Michelin',           'market': 'cac40', 'currency': 'EUR'},
    'OR.PA':    {'name': "L'Oréal",            'market': 'cac40', 'currency': 'EUR'},
    'ORA.PA':   {'name': 'Orange',             'market': 'cac40', 'currency': 'EUR'},
    'PUB.PA':   {'name': 'Publicis',           'market': 'cac40', 'currency': 'EUR'},
    'RMS.PA':   {'name': 'Hermès',             'market': 'cac40', 'currency': 'EUR'},
    'RNO.PA':   {'name': 'Renault',            'market': 'cac40', 'currency': 'EUR'},
    'SAF.PA':   {'name': 'Safran',             'market': 'cac40', 'currency': 'EUR'},
    'SAN.PA':   {'name': 'Sanofi',             'market': 'cac40', 'currency': 'EUR'},
    'SGO.PA':   {'name': 'Saint-Gobain',       'market': 'cac40', 'currency': 'EUR'},
    'STMPA.PA': {'name': 'STMicroelectronics', 'market': 'cac40', 'currency': 'EUR'},
    'SU.PA':    {'name': 'Schneider Electric', 'market': 'cac40', 'currency': 'EUR'},
    'TEP.PA':   {'name': 'Teleperformance',    'market': 'cac40', 'currency': 'EUR'},
    'TTE.PA':   {'name': 'TotalEnergies',      'market': 'cac40', 'currency': 'EUR'},
    'URW.PA':   {'name': 'Unibail-Rodamco',    'market': 'cac40', 'currency': 'EUR'},
    'VIE.PA':   {'name': 'Veolia',             'market': 'cac40', 'currency': 'EUR'},
    'VIV.PA':   {'name': 'Vivendi',            'market': 'cac40', 'currency': 'EUR'},
    'WLN.PA':   {'name': 'Worldline',          'market': 'cac40', 'currency': 'EUR'},
    'FR.PA':    {'name': 'Valeo',              'market': 'cac40', 'currency': 'EUR'},
}

DOW30 = {
    'AAPL': {'name': 'Apple',              'market': 'dow30', 'currency': 'USD'},
    'AMGN': {'name': 'Amgen',              'market': 'dow30', 'currency': 'USD'},
    'AXP':  {'name': 'American Express',   'market': 'dow30', 'currency': 'USD'},
    'BA':   {'name': 'Boeing',             'market': 'dow30', 'currency': 'USD'},
    'CAT':  {'name': 'Caterpillar',        'market': 'dow30', 'currency': 'USD'},
    'CRM':  {'name': 'Salesforce',         'market': 'dow30', 'currency': 'USD'},
    'CSCO': {'name': 'Cisco',              'market': 'dow30', 'currency': 'USD'},
    'CVX':  {'name': 'Chevron',            'market': 'dow30', 'currency': 'USD'},
    'DIS':  {'name': 'Disney',             'market': 'dow30', 'currency': 'USD'},
    'DOW':  {'name': 'Dow Inc.',           'market': 'dow30', 'currency': 'USD'},
    'GS':   {'name': 'Goldman Sachs',      'market': 'dow30', 'currency': 'USD'},
    'HD':   {'name': 'Home Depot',         'market': 'dow30', 'currency': 'USD'},
    'HON':  {'name': 'Honeywell',          'market': 'dow30', 'currency': 'USD'},
    'IBM':  {'name': 'IBM',                'market': 'dow30', 'currency': 'USD'},
    'INTC': {'name': 'Intel',              'market': 'dow30', 'currency': 'USD'},
    'JNJ':  {'name': 'Johnson & Johnson',  'market': 'dow30', 'currency': 'USD'},
    'JPM':  {'name': 'JPMorgan Chase',     'market': 'dow30', 'currency': 'USD'},
    'KO':   {'name': 'Coca-Cola',          'market': 'dow30', 'currency': 'USD'},
    'MCD':  {'name': "McDonald's",         'market': 'dow30', 'currency': 'USD'},
    'MMM':  {'name': '3M',                 'market': 'dow30', 'currency': 'USD'},
    'MRK':  {'name': 'Merck',              'market': 'dow30', 'currency': 'USD'},
    'MSFT': {'name': 'Microsoft',          'market': 'dow30', 'currency': 'USD'},
    'NKE':  {'name': 'Nike',               'market': 'dow30', 'currency': 'USD'},
    'PG':   {'name': 'Procter & Gamble',   'market': 'dow30', 'currency': 'USD'},
    'TRV':  {'name': 'Travelers',          'market': 'dow30', 'currency': 'USD'},
    'UNH':  {'name': 'UnitedHealth',       'market': 'dow30', 'currency': 'USD'},
    'V':    {'name': 'Visa',               'market': 'dow30', 'currency': 'USD'},
    'VZ':   {'name': 'Verizon',            'market': 'dow30', 'currency': 'USD'},
    'WBA':  {'name': 'Walgreens',          'market': 'dow30', 'currency': 'USD'},
    'WMT':  {'name': 'Walmart',            'market': 'dow30', 'currency': 'USD'},
}

ALL_ASSETS = {**ETFS, **CAC40, **DOW30}
HORIZONS = [1, 3, 5, 10]

# ═══════════════════════════════════════════════
# INDICATEURS TECHNIQUES
# ═══════════════════════════════════════════════

def compute_features(df):
    for d in [1, 3, 5, 10, 20]:
        df[f'return_{d}d'] = df['Close'].pct_change(d)
    for w in [5, 10, 20, 50]:
        df[f'ma{w}'] = df['Close'].rolling(w).mean()
        df[f'price_ma{w}_ratio'] = df['Close'] / df[f'ma{w}']
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold']   = (df['rsi'] < 30).astype(int)
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff']   = df['macd'] - df['macd_signal']
    df['macd_cross']  = (df['macd_diff'] > 0).astype(int)
    df['volatility_5d']  = df['return_1d'].rolling(5).std()
    df['volatility_20d'] = df['return_1d'].rolling(20).std()
    df['bb_upper'] = df['ma20'] + 2 * df['volatility_20d']
    df['bb_lower'] = df['ma20'] - 2 * df['volatility_20d']
    df['bb_pos']   = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['vol_ma10']     = df['Volume'].rolling(10).mean()
        df['volume_ratio'] = df['Volume'] / df['vol_ma10']
    else:
        df['volume_ratio'] = 1.0
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    return df

FEATURE_COLS = [
    'return_1d','return_3d','return_5d','return_10d','return_20d',
    'price_ma5_ratio','price_ma10_ratio','price_ma20_ratio','price_ma50_ratio',
    'rsi','rsi_overbought','rsi_oversold',
    'macd','macd_signal','macd_diff','macd_cross',
    'volatility_5d','volatility_20d','bb_pos','volume_ratio',
    'momentum_10','momentum_20',
]

# ═══════════════════════════════════════════════
# ENTRAÎNEMENT
# ═══════════════════════════════════════════════

def train_asset(ticker, meta):
    df = yf.download(ticker, period='5y', interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 200:
        return None
    df = compute_features(df)
    result = {
        'ticker':    ticker,
        'name':      meta['name'],
        'market':    meta['market'],
        'currency':  meta['currency'],
        'price':     round(float(df['Close'].iloc[-1]), 2),
        'change_1d': round(float(df['return_1d'].iloc[-1]) * 100, 2),
        'horizons':  {}
    }
    last_features = None
    for h in HORIZONS:
        df[f'target_{h}d'] = (df['Close'].shift(-h) > df['Close']).astype(int)
        data = df[FEATURE_COLS + [f'target_{h}d']].dropna()
        if len(data) < 100:
            continue
        X, y = data[FEATURE_COLS], data[f'target_{h}d']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=8, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if last_features is None:
            last_features = X.iloc[-1:].values
        proba = model.predict_proba(last_features)[0]
        pred  = int(model.predict(last_features)[0])
        result['horizons'][f'{h}d'] = {
            'prediction': 'up' if pred == 1 else 'down',
            'confidence': round(float(max(proba)) * 100, 1),
            'accuracy':   round(acc * 100, 1),
            'signal':     '🟢' if pred == 1 else '🔴',
        }
    return result if result['horizons'] else None

# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def predict_all():
    print(f"\n🤖 WSPP IA — {len(ALL_ASSETS)} actifs · Horizons 1j 3j 5j 10j\n")
    results = {'etf': {}, 'cac40': {}, 'dow30': {}}
    total = len(ALL_ASSETS)
    for i, (ticker, meta) in enumerate(ALL_ASSETS.items(), 1):
        print(f"[{i:3}/{total}] {ticker:12} {meta['name'][:25]:<25}", end=' ', flush=True)
        try:
            r = train_asset(ticker, meta)
            if r:
                results[meta['market']][ticker] = r
                sig = r['horizons'].get('5d', {})
                print(f"{sig.get('signal','?')} {sig.get('prediction','?'):5} conf:{sig.get('confidence',0)}% acc:{sig.get('accuracy',0)}%")
            else:
                print("⚠️  skip")
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
    return results

if __name__ == '__main__':
    results = predict_all()

    # Top signaux
    print("\n" + "="*65)
    all_items = []
    for market, assets in results.items():
        for ticker, r in assets.items():
            h = r['horizons'].get('5d')
            if h:
                all_items.append((r['name'], ticker, r['market'], h))
    all_items.sort(key=lambda x: x[3]['confidence'], reverse=True)

    print("🟢 TOP 10 OPPORTUNITÉS HAUSSE (5 jours)")
    n = 0
    for name, ticker, market, h in all_items:
        if h['prediction'] == 'up' and n < 10:
            print(f"  {ticker:12} {name[:25]:<25} {h['confidence']}% confiance · {h['accuracy']}% précision")
            n += 1

    print("\n🔴 TOP 10 TITRES À ÉVITER (5 jours)")
    n = 0
    for name, ticker, market, h in reversed(all_items):
        if h['prediction'] == 'down' and n < 10:
            print(f"  {ticker:12} {name[:25]:<25} {h['confidence']}% confiance · {h['accuracy']}% précision")
            n += 1

    with open('predictions.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    total_done = sum(len(v) for v in results.values())
    print(f"\n✅ {total_done}/{len(ALL_ASSETS)} actifs analysés → predictions.json")
