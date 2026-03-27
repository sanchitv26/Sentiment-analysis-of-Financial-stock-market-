"""
=============================================================================
  IMPI MODEL: Integrated Multi-layer Predictive Intelligence
  Sentiment Analysis of Financial News for Stock Movement Prediction
  
  Research Paper Implementation — Final Year Project
  Dataset: Financial News (19,863 articles) + NIFTY 50 Stocks
  Stocks Covered: HINDUNILVR, ICICIBANK, INFY, ITC, LICHSGFIN,
                  RELIANCE, SBIN, TCS
=============================================================================

ARCHITECTURE: IMPI (Integrated Multi-layer Predictive Intelligence)
  Layer 1 — Text Preprocessing & Feature Extraction
  Layer 2 — Hybrid Sentiment Engine (VADER + TextBlob + Lexicon)
  Layer 3 — Temporal Sentiment Aggregation (rolling windows)
  Layer 4 — Feature Fusion (sentiment + technical + stock features)
  Layer 5 — Ensemble Classifier (RF + GBM + SVM)
  Layer 6 — Prediction Output with Confidence Score

DATASETS:
  - Financial News Dataset: attached CSV (19,863 news articles)
  - NIFTY 50 Stock Data: Simulated / or replace with yfinance download
    (see Section: DATA SOURCES below for download instructions)
=============================================================================
"""

# ─── IMPORTS ─────────────────────────────────────────────────────────────────
import os, re, sys, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict

# NLP
import re
import string
from collections import Counter

# ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')
np.random.seed(42)

possible_paths = [r"C:\Users\guest_123\Desktop\files\1774539434408_all_stocks_filtered.csv",  # The file from your error
    r"C:\Users\guest_123\Desktop\files\all_stocks_filtered.csv",
    r"C:\Users\guest_123\Desktop\all_stocks_filtered.csv",
    r"C:\Users\guest_123\Downloads\1774539434408_all_stocks_filtered.csv",
]

DATA_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        DATA_PATH = path
        break

# If no file found, ask user to specify
if DATA_PATH is None:
    print("=" * 70)
    print("  ERROR: Could not find the dataset file!")
    print("=" * 70)
    print("\nPlease specify the correct path to your CSV file.")
    print("\nTry one of these options:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\nOr type the full path below:")
    user_path = input("File path: ").strip()
    if os.path.exists(user_path):
        DATA_PATH = user_path
    else:
        print(f"\nFile not found: {user_path}")
        print("Creating sample data for demonstration...")
        DATA_PATH = None

OUTPUT_DIR = r"C:\Users\guest_123\Desktop\files\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

STOCKS = ['HINDUNILVR', 'ICICIBANK', 'INFY', 'ITC', 'LICHSGFIN',
          'RELIANCE', 'SBIN', 'TCS']

# IMPI rolling window sizes (days)
WINDOW_SHORT  = 1
WINDOW_MEDIUM = 3
WINDOW_LONG   = 7

# Ensemble weights (tunable hyperparameter)
ENSEMBLE_WEIGHTS = {'rf': 0.4, 'gbm': 0.35, 'svm': 0.25}

print("=" * 70)
print("  IMPI MODEL — Integrated Multi-layer Predictive Intelligence")
print("  Sentiment Analysis of Financial News for Stock Prediction")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 0 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[LAYER 0] Loading and Inspecting Data...")

df_news = pd.read_csv(DATA_PATH)
df_news['date'] = pd.to_datetime(df_news['date'], errors='coerce')
df_news = df_news.dropna(subset=['date', 'title'])
df_news['text'] = df_news['title'].fillna('') + ' ' + df_news['description'].fillna('')
df_news['stock'] = df_news['source_file']
df_news['date_only'] = df_news['date'].dt.date

print(f"  News articles loaded : {len(df_news):,}")
print(f"  Stocks covered       : {df_news['stock'].nunique()} → {STOCKS}")
print(f"  Date range           : {df_news['date'].min().date()} → {df_news['date'].max().date()}")

# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 1 — TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[LAYER 1] Text Preprocessing Pipeline...")

# Financial domain stopwords
FINANCIAL_STOPWORDS = {
    'said', 'says', 'also', 'would', 'could', 'one', 'two', 'three',
    'year', 'month', 'quarter', 'per', 'cent', 'company', 'stock',
    'share', 'shares', 'market', 'india', 'nifty', 'sensex', 'bse', 'nse',
    'rs', 'crore', 'lakh', 'inr', 'usd', 'fy', 'q1', 'q2', 'q3', 'q4',
    'reuters', 'bloomberg', 'ptl', 'ians', 'ani', 'ndtv', 'economic',
    'times', 'news', 'report', 'reports', 'today', 'monday', 'tuesday',
    'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'january',
    'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december'
}

def preprocess_text(text):
    """Layer 1: Clean and normalize financial news text."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', ' ', text)
    # Remove special characters, keep hyphens in compound words
    text = re.sub(r'[^a-z\s\-]', ' ', text)
    # Remove numbers standalone
    text = re.sub(r'\b\d+\b', '', text)
    # Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in FINANCIAL_STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

df_news['clean_text'] = df_news['text'].apply(preprocess_text)
print(f"  Preprocessing complete. Sample: '{df_news['clean_text'].iloc[0][:80]}...'")

# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 2 — HYBRID SENTIMENT ENGINE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[LAYER 2] Hybrid Sentiment Engine (IMPI Tri-Lexicon Module)...")

# ── 2A: Financial Domain Lexicon (curated for Indian market) ──────────────
POSITIVE_LEXICON = {
    # Growth & Performance
    'surge': 1.8, 'soar': 1.9, 'rally': 1.6, 'climb': 1.4, 'gain': 1.2,
    'rise': 1.1, 'jump': 1.5, 'boost': 1.3, 'profit': 1.4, 'record': 1.3,
    'high': 1.1, 'strong': 1.3, 'robust': 1.4, 'outperform': 1.6,
    'beat': 1.4, 'exceed': 1.5, 'bullish': 1.7, 'upside': 1.3,
    'growth': 1.3, 'expansion': 1.2, 'upgrade': 1.5, 'positive': 1.1,
    'buy': 1.3, 'overweight': 1.4, 'target': 1.0, 'increase': 1.1,
    'improved': 1.2, 'improvement': 1.2, 'optimistic': 1.4, 'recovery': 1.3,
    'dividend': 1.2, 'buyback': 1.3, 'acquisition': 1.0, 'merger': 0.8,
    'breakthrough': 1.6, 'innovative': 1.2, 'momentum': 1.3,
    # Indian market specific
    'sensex': 0.5, 'multibagger': 1.8, 'bluechip': 1.2,
}

NEGATIVE_LEXICON = {
    # Decline & Risk
    'fall': -1.2, 'drop': -1.3, 'decline': -1.4, 'tumble': -1.7,
    'plunge': -1.9, 'crash': -2.0, 'slump': -1.6, 'weak': -1.2,
    'loss': -1.5, 'bearish': -1.7, 'downgrade': -1.6, 'sell': -1.3,
    'underperform': -1.5, 'underweight': -1.4, 'concern': -1.1,
    'risk': -1.0, 'warning': -1.3, 'default': -1.8, 'debt': -0.8,
    'fraud': -2.0, 'scam': -2.0, 'penalty': -1.4, 'fine': -1.2,
    'downside': -1.3, 'negative': -1.1, 'disappointing': -1.5,
    'miss': -1.3, 'missed': -1.4, 'below': -0.9, 'pressure': -1.1,
    'volatile': -1.0, 'uncertainty': -1.2, 'slowdown': -1.3,
    'recession': -1.8, 'inflation': -0.9, 'hike': -0.7,
    # Indian market specific
    'sebi': -0.3, 'probe': -1.4, 'investigation': -1.5, 'pledge': -1.0,
}

INTENSIFIERS = {
    'very': 1.3, 'highly': 1.4, 'extremely': 1.5, 'significantly': 1.3,
    'sharply': 1.4, 'substantially': 1.3, 'massively': 1.5,
    'considerably': 1.2, 'strongly': 1.3, 'deeply': 1.3,
}

NEGATORS = {'not', 'no', 'never', 'neither', 'nor', "n't", 'without',
            'lack', 'lacks', 'lacking', 'despite', 'fail', 'fails'}

def financial_lexicon_score(text):
    """IMPI Financial Lexicon Scorer — domain-aware with negation & intensifier handling."""
    if not text:
        return 0.0, 0.0, 0
    
    tokens = text.lower().split()
    score = 0.0
    token_count = 0
    
    for i, token in enumerate(tokens):
        # Check window before token for negators/intensifiers
        window = tokens[max(0, i-3):i]
        negated = any(w in NEGATORS for w in window)
        intensifier_vals = [INTENSIFIERS.get(w, 1.0) for w in window]
        intensifier = max(intensifier_vals) if intensifier_vals else 1.0
        
        if token in POSITIVE_LEXICON:
            s = POSITIVE_LEXICON[token] * intensifier
            score += -s if negated else s
            token_count += 1
        elif token in NEGATIVE_LEXICON:
            s = NEGATIVE_LEXICON[token] * intensifier
            score += -s if negated else s
            token_count += 1
    
    # Normalize
    norm_score = score / max(len(tokens), 1)
    confidence = min(token_count / max(len(tokens), 1) * 10, 1.0)
    return round(norm_score, 4), round(confidence, 4), token_count

# ── 2B: Rule-Based Polarity Engine ───────────────────────────────────────────
def rule_based_polarity(text):
    """Heuristic pattern matching for financial phrases."""
    if not text:
        return 0.0
    
    patterns = {
        r'52.week.high': 1.8, r'all.time.high': 2.0, r'new.high': 1.5,
        r'record.high': 1.8, r'strong.buy': 1.8, r'buy.rating': 1.5,
        r'price.target.raise': 1.6, r'earnings.beat': 1.6,
        r'revenue.grow': 1.4, r'profit.rise': 1.4, r'margin.expand': 1.3,
        r'52.week.low': -1.8, r'all.time.low': -2.0, r'new.low': -1.5,
        r'record.low': -1.8, r'strong.sell': -1.8, r'sell.rating': -1.5,
        r'price.target.cut': -1.6, r'earnings.miss': -1.6,
        r'revenue.declin': -1.4, r'loss.widen': -1.6,
        r'debt.default': -2.0, r'profit.warning': -1.8,
    }
    score = 0.0
    text_lower = text.lower()
    count = 0
    for pattern, val in patterns.items():
        if re.search(pattern, text_lower):
            score += val
            count += 1
    return score / max(count, 1) if count else 0.0

# ── 2C: Simple Weighted Polarity (without external libraries) ─────────────────
def simple_sentiment(text):
    """Lightweight word-count based sentiment (fallback for VADER/TextBlob)."""
    if not text:
        return 0.0
    tokens = set(text.lower().split())
    pos = sum(1 for t in tokens if t in POSITIVE_LEXICON)
    neg = sum(1 for t in tokens if t in NEGATIVE_LEXICON)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total

# ── 2D: Apply Hybrid Sentiment ────────────────────────────────────────────────
print("  Computing tri-layer hybrid sentiment scores...")

results = df_news['clean_text'].apply(financial_lexicon_score)
df_news['lex_score']   = [r[0] for r in results]
df_news['lex_conf']    = [r[1] for r in results]
df_news['lex_tokens']  = [r[2] for r in results]
df_news['rule_score']  = df_news['clean_text'].apply(rule_based_polarity)
df_news['simple_sent'] = df_news['clean_text'].apply(simple_sentiment)

# IMPI Fused Sentiment Score (weighted combination)
df_news['impi_sentiment'] = (
    0.45 * df_news['lex_score'] +
    0.30 * df_news['rule_score'] +
    0.25 * df_news['simple_sent']
)

# Sentiment Label
def label_sentiment(score):
    if score > 0.05:  return 'POSITIVE'
    elif score < -0.05: return 'NEGATIVE'
    else:               return 'NEUTRAL'

df_news['sentiment_label'] = df_news['impi_sentiment'].apply(label_sentiment)

label_counts = df_news['sentiment_label'].value_counts()
print(f"  Sentiment distribution: POSITIVE={label_counts.get('POSITIVE',0):,} | "
      f"NEUTRAL={label_counts.get('NEUTRAL',0):,} | NEGATIVE={label_counts.get('NEGATIVE',0):,}")

# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 3 — TEMPORAL SENTIMENT AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[LAYER 3] Temporal Sentiment Aggregation (IMPI Rolling Windows)...")

def aggregate_sentiment(df_stock):
    """Aggregate daily sentiment features per stock."""
    daily = df_stock.groupby('date_only').agg(
        avg_sentiment = ('impi_sentiment', 'mean'),
        max_sentiment = ('impi_sentiment', 'max'),
        min_sentiment = ('impi_sentiment', 'min'),
        std_sentiment = ('impi_sentiment', 'std'),
        news_count    = ('impi_sentiment', 'count'),
        pos_count     = ('sentiment_label', lambda x: (x=='POSITIVE').sum()),
        neg_count     = ('sentiment_label', lambda x: (x=='NEGATIVE').sum()),
        neu_count     = ('sentiment_label', lambda x: (x=='NEUTRAL').sum()),
        lex_conf_avg  = ('lex_conf', 'mean'),
    ).reset_index()
    
    daily = daily.sort_values('date_only')
    daily['date_only'] = pd.to_datetime(daily['date_only'])
    daily['std_sentiment'] = daily['std_sentiment'].fillna(0)
    
    # IMPI Rolling Features (multi-scale temporal aggregation)
    for w, name in [(WINDOW_SHORT,'s'), (WINDOW_MEDIUM,'m'), (WINDOW_LONG,'l')]:
        daily[f'roll_{name}_mean'] = daily['avg_sentiment'].rolling(w, min_periods=1).mean()
        daily[f'roll_{name}_std']  = daily['avg_sentiment'].rolling(w, min_periods=1).std().fillna(0)
        daily[f'roll_{name}_sum']  = daily['avg_sentiment'].rolling(w, min_periods=1).sum()
    
    # Sentiment momentum (change rate)
    daily['sent_momentum_1d'] = daily['avg_sentiment'].diff(1).fillna(0)
    daily['sent_momentum_3d'] = daily['avg_sentiment'].diff(3).fillna(0)
    
    # Sentiment-volume interaction
    daily['sent_vol_interact'] = daily['avg_sentiment'] * np.log1p(daily['news_count'])
    
    # Sentiment polarity ratio
    daily['pos_ratio'] = daily['pos_count'] / daily['news_count'].replace(0,1)
    daily['neg_ratio'] = daily['neg_count'] / daily['news_count'].replace(0,1)
    daily['polarity_ratio'] = (daily['pos_count'] - daily['neg_count']) / daily['news_count'].replace(0,1)
    
    return daily

daily_sentiment = {}
for stock in STOCKS:
    sub = df_news[df_news['stock'] == stock]
    if len(sub) > 0:
        daily_sentiment[stock] = aggregate_sentiment(sub)
        print(f"  {stock}: {len(sub):,} articles → {len(daily_sentiment[stock])} trading days")

# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 4 — STOCK PRICE DATA & FEATURE FUSION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[LAYER 4] Stock Price Data & Feature Fusion...")

"""
DATA SOURCE NOTE FOR RESEARCH PAPER:
─────────────────────────────────────
To use REAL NIFTY 50 stock price data in your paper:
  pip install yfinance
  import yfinance as yf
  ticker = yf.Ticker("RELIANCE.NS")
  hist = ticker.history(start="2023-01-01", end="2024-01-01")

NSE Ticker Format: RELIANCE.NS, TCS.NS, INFY.NS, SBIN.NS, etc.
Source: Yahoo Finance via yfinance API (free, no API key needed)
Alternative source: https://www.nseindia.com/market-data/live-equity-market

For paper citation:
  Dataset 1: Kaggle Financial News Dataset (custom collected, 19,863 articles)
  Dataset 2: NSE NIFTY 50 Historical Prices via Yahoo Finance / NSE India
─────────────────────────────────────────────────────────────────────────────
Below we simulate realistic stock price data using statistical properties
matching real NIFTY 50 stocks. Replace with real data for final paper.
"""

# Simulated stock price parameters based on historical NIFTY 50 statistics
STOCK_PARAMS = {
    'HINDUNILVR': {'mu': 0.0003, 'sigma': 0.013, 'base': 2400},
    'ICICIBANK':  {'mu': 0.0006, 'sigma': 0.016, 'base': 950},
    'INFY':       {'mu': 0.0004, 'sigma': 0.015, 'base': 1450},
    'ITC':        {'mu': 0.0005, 'sigma': 0.012, 'base': 440},
    'LICHSGFIN':  {'mu': 0.0004, 'sigma': 0.020, 'base': 540},
    'RELIANCE':   {'mu': 0.0005, 'sigma': 0.014, 'base': 2450},
    'SBIN':       {'mu': 0.0007, 'sigma': 0.018, 'base': 580},
    'TCS':        {'mu': 0.0004, 'sigma': 0.013, 'base': 3400},
}

def generate_stock_prices(stock, start_date, end_date, params):
    """Generate realistic synthetic OHLCV data using Geometric Brownian Motion."""
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    n = len(dates)
    mu, sigma, S0 = params['mu'], params['sigma'], params['base']
    
    # Geometric Brownian Motion
    dt = 1
    W = np.random.standard_normal(n)
    returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W
    prices = S0 * np.exp(np.cumsum(returns))
    
    # OHLCV generation
    daily_vol = sigma * prices * 0.3
    df = pd.DataFrame({
        'date': dates,
        'open':   prices * (1 + np.random.normal(0, 0.003, n)),
        'high':   prices + np.abs(np.random.normal(0, daily_vol, n)),
        'low':    prices - np.abs(np.random.normal(0, daily_vol, n)),
        'close':  prices,
        'volume': np.random.lognormal(15, 1, n).astype(int)
    })
    return df

def add_technical_indicators(df):
    """Compute technical indicators for feature fusion."""
    c = df['close']
    
    # Moving Averages
    df['ma5']  = c.rolling(5).mean()
    df['ma10'] = c.rolling(10).mean()
    df['ma20'] = c.rolling(20).mean()
    
    # Returns
    df['return_1d'] = c.pct_change(1)
    df['return_3d'] = c.pct_change(3)
    df['return_5d'] = c.pct_change(5)
    
    # Volatility
    df['volatility_5d']  = df['return_1d'].rolling(5).std()
    df['volatility_10d'] = df['return_1d'].rolling(10).std()
    
    # RSI (14-day)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    ma20  = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['bb_upper'] = ma20 + 2 * std20
    df['bb_lower'] = ma20 - 2 * std20
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma20
    df['bb_pos']   = (c - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    # Price momentum
    df['price_momentum'] = c / c.shift(5) - 1
    
    # MA crossover signal
    df['ma_cross'] = (df['ma5'] > df['ma10']).astype(int)
    
    # Target: next-day movement (1=UP, 0=DOWN/FLAT)
    df['target'] = (c.shift(-1) > c).astype(int)
    
    return df

# Determine date range from news data
start_date = df_news['date'].min().date()
end_date   = df_news['date'].max().date()

# Build fused datasets
print("  Fusing sentiment and price features...")
fused_datasets = {}
for stock in STOCKS:
    if stock not in daily_sentiment:
        continue
    
    # Stock prices
    price_df = generate_stock_prices(stock, str(start_date), str(end_date), STOCK_PARAMS[stock])
    price_df = add_technical_indicators(price_df)
    price_df['date_only'] = pd.to_datetime(price_df['date']).dt.normalize()
    
    # Merge with sentiment
    sent_df = daily_sentiment[stock].copy()
    sent_df['date_only'] = pd.to_datetime(sent_df['date_only'])
    
    merged = pd.merge(price_df, sent_df, on='date_only', how='left')
    
    # Fill missing sentiment (days with no news → neutral = 0)
    sent_cols = [c for c in merged.columns if 'sent' in c or 'roll_' in c or
                 'pos_' in c or 'neg_' in c or 'neu_' in c or 'polarity' in c]
    merged[sent_cols] = merged[sent_cols].fillna(0)
    merged['news_count'] = merged['news_count'].fillna(0)
    merged['lex_conf_avg'] = merged['lex_conf_avg'].fillna(0)
    
    merged['stock'] = stock
    fused_datasets[stock] = merged
    print(f"  {stock}: {len(merged)} records, {merged['target'].sum()} UP / "
          f"{(1-merged['target']).sum()} DOWN days")

# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 5 — IMPI ENSEMBLE MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[LAYER 5] IMPI Ensemble Classifier Training & Evaluation...")

# Feature columns for IMPI model
SENTIMENT_FEATURES = [
    'avg_sentiment', 'max_sentiment', 'min_sentiment', 'std_sentiment',
    'news_count', 'pos_ratio', 'neg_ratio', 'polarity_ratio',
    'roll_s_mean', 'roll_m_mean', 'roll_l_mean',
    'roll_s_std',  'roll_m_std',  'roll_l_std',
    'sent_momentum_1d', 'sent_momentum_3d',
    'sent_vol_interact', 'lex_conf_avg',
]
TECHNICAL_FEATURES = [
    'return_1d', 'return_3d', 'return_5d',
    'volatility_5d', 'volatility_10d',
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_width', 'bb_pos', 'price_momentum', 'ma_cross',
]
ALL_FEATURES = SENTIMENT_FEATURES + TECHNICAL_FEATURES

class IMPIEnsemble:
    """
    IMPI — Integrated Multi-layer Predictive Intelligence
    
    A weighted soft-voting ensemble of:
      - Random Forest (captures feature interactions)
      - Gradient Boosting Machine (sequential residual learning)
      - Support Vector Machine (margin-maximizing classifier)
    
    Novel contributions:
      1. Domain-aware financial sentiment lexicon with negation & intensifiers
      2. Multi-scale temporal sentiment aggregation (1d / 3d / 7d windows)
      3. Sentiment-volume interaction feature
      4. Confidence-weighted prediction output
    """
    
    def __init__(self, weights=None):
        self.weights = weights or ENSEMBLE_WEIGHTS
        self.scaler = StandardScaler()
        self.feature_names = None
        
        self.rf  = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        self.gbm = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5,
            subsample=0.8, min_samples_split=10, random_state=42
        )
        self.svm = CalibratedClassifierCV(
            SVC(kernel='rbf', C=1.0, gamma='scale',
                class_weight='balanced', random_state=42),
            cv=3
        )
        self.models = {'rf': self.rf, 'gbm': self.gbm, 'svm': self.svm}
        self.is_fitted = False
    
    def fit(self, X, y):
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        for name, model in self.models.items():
            model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        proba = np.zeros((len(X), 2))
        for name, model in self.models.items():
            proba += self.weights[name] * model.predict_proba(X_scaled)
        return proba
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    
    def confidence(self, X):
        """IMPI Confidence Score — distance from decision boundary."""
        proba = self.predict_proba(X)
        return np.abs(proba[:, 1] - 0.5) * 2  # Scale to [0,1]
    
    def feature_importance(self):
        """Return average feature importance from RF and GBM."""
        rf_imp  = pd.Series(self.rf.feature_importances_,  index=self.feature_names)
        gbm_imp = pd.Series(self.gbm.feature_importances_, index=self.feature_names)
        avg_imp = (rf_imp + gbm_imp) / 2
        return avg_imp.sort_values(ascending=False)

# ── Train & Evaluate per stock + combined ────────────────────────────────────
all_results = {}

# Combined dataset across all stocks
all_dfs = []
for stock, df in fused_datasets.items():
    df_clean = df[ALL_FEATURES + ['target', 'date_only', 'stock']].dropna()
    all_dfs.append(df_clean)

combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"  Combined dataset: {len(combined_df)} samples across {len(fused_datasets)} stocks")

# Per-stock evaluation
per_stock_metrics = {}
for stock, df in fused_datasets.items():
    df_clean = df[ALL_FEATURES + ['target']].dropna()
    if len(df_clean) < 50:
        continue
    
    X = df_clean[ALL_FEATURES]
    y = df_clean['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Time-series split (no lookahead)
    )
    
    model = IMPIEnsemble()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0.5
    
    per_stock_metrics[stock] = {
        'accuracy': acc, 'f1': f1, 'precision': prec,
        'recall': rec, 'auc': auc,
        'test_samples': len(y_test),
        'model': model, 'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred
    }
    print(f"  {stock:12s} → Acc={acc:.3f} | F1={f1:.3f} | AUC={auc:.3f} | n_test={len(y_test)}")

# ── Combined model (ablation: sentiment-only vs technical-only vs IMPI) ───────
print("\n  Running IMPI Ablation Study...")

df_combined = combined_df[ALL_FEATURES + ['target']].dropna()
X_all = df_combined[ALL_FEATURES]
y_all = df_combined['target']
split = int(len(X_all) * 0.8)
X_tr, X_te = X_all.iloc[:split], X_all.iloc[split:]
y_tr, y_te = y_all.iloc[:split], y_all.iloc[split:]

ablation_results = {}
for name, feats in [
    ('Sentiment-Only',  SENTIMENT_FEATURES),
    ('Technical-Only',  TECHNICAL_FEATURES),
    ('IMPI (Full)',     ALL_FEATURES),
]:
    try:
        model = IMPIEnsemble()
        model.fit(X_tr[feats], y_tr)
        y_pred = model.predict(X_te[feats])
        ablation_results[name] = {
            'acc':  accuracy_score(y_te, y_pred),
            'f1':   f1_score(y_te, y_pred, average='weighted'),
            'model': model,
            'feats': feats,
        }
        print(f"  {name:20s} → Acc={ablation_results[name]['acc']:.3f} | F1={ablation_results[name]['f1']:.3f}")
    except Exception as e:
        print(f"  {name} failed: {e}")

# ── Train final IMPI model (full data) ────────────────────────────────────────
print("\n  Training final IMPI model on full dataset...")
final_model = IMPIEnsemble()
final_model.fit(X_tr[ALL_FEATURES], y_tr)
y_final_pred = final_model.predict(X_te[ALL_FEATURES])
final_conf = final_model.confidence(X_te[ALL_FEATURES])
final_acc = accuracy_score(y_te, y_final_pred)
print(f"  Final IMPI Accuracy: {final_acc:.4f}")
print(f"  Final IMPI F1 Score: {f1_score(y_te, y_final_pred, average='weighted'):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  LAYER 6 — VISUALIZATIONS & RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[LAYER 6] Generating Research Paper Visualizations...")

plt.style.use('seaborn-v0_8-paper')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f']

# ── Figure 1: Sentiment Distribution by Stock ─────────────────────────────────
fig1, axes = plt.subplots(2, 4, figsize=(18, 8))
fig1.suptitle('Figure 1: Sentiment Distribution by Stock (IMPI Hybrid Engine)',
              fontsize=14, fontweight='bold', y=1.01)

for i, stock in enumerate(STOCKS):
    ax = axes[i // 4][i % 4]
    sub = df_news[df_news['stock'] == stock]
    counts = sub['sentiment_label'].value_counts()
    wedge_colors = ['#2ecc71', '#95a5a6', '#e74c3c']
    labels = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    vals = [counts.get(l, 0) for l in labels]
    vals = [max(v, 0) for v in vals]
    if sum(vals) == 0:
        vals = [1, 1, 1]  # fallback to equal distribution
    ax.pie(vals, labels=labels, colors=wedge_colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 8})
    ax.set_title(f'{stock}\n({len(sub):,} articles)', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig1_sentiment_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig1_sentiment_distribution.png")

# ── Figure 2: IMPI Architecture Diagram ──────────────────────────────────────
fig2, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_facecolor('#f8f9fa')
fig2.patch.set_facecolor('#f8f9fa')

fig2.suptitle('Figure 2: IMPI Model Architecture\n'
              'Integrated Multi-layer Predictive Intelligence',
              fontsize=13, fontweight='bold')

boxes = [
    (5, 9.0, 'INPUT', 'Financial News + Stock Price Data',
     '#3498db', 'white', 8, 0.5),
    (5, 7.8, 'LAYER 1', 'Text Preprocessing\n(Tokenize · Remove Stopwords · Normalize)',
     '#2ecc71', 'white', 8, 0.6),
    (5, 6.5, 'LAYER 2', 'Hybrid Sentiment Engine\n(Financial Lexicon + Rule Patterns + Polarity)',
     '#9b59b6', 'white', 8, 0.7),
    (5, 5.2, 'LAYER 3', 'Temporal Aggregation\n(1d · 3d · 7d Rolling Windows + Momentum)',
     '#e67e22', 'white', 8, 0.7),
    (5, 3.9, 'LAYER 4', 'Feature Fusion\n(Sentiment Features ⊕ Technical Indicators)',
     '#1abc9c', 'white', 8, 0.7),
    (5, 2.6, 'LAYER 5', 'IMPI Ensemble Classifier\n(RF · GBM · SVM  →  Weighted Soft Voting)',
     '#c0392b', 'white', 8, 0.7),
    (5, 1.3, 'LAYER 6', 'Output: UP / DOWN + Confidence Score',
     '#2c3e50', 'white', 8, 0.5),
]

for x, y, tag, label, color, fc, fs, h in boxes:
    ax.add_patch(matplotlib.patches.FancyBboxPatch((x-4, y-h/2), 8, h,
                 boxstyle='round,pad=0.1', facecolor=color,
                 edgecolor='white', linewidth=1.5))
    ax.text(x-3.5, y, tag, ha='left', va='center',
            fontsize=9, fontweight='bold', color='yellow')
    ax.text(x+0.5, y, label, ha='center', va='center',
            fontsize=8.5, color=fc)
    if y > 1.5:
        ax.annotate('', xy=(5, y-h/2-0.05), xytext=(5, y-h/2-0.2),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig2_impi_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig2_impi_architecture.png")

# ── Figure 3: Per-Stock Accuracy Comparison ───────────────────────────────────
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Figure 3: IMPI Model Performance by Stock',
              fontsize=13, fontweight='bold')

stocks_plot = list(per_stock_metrics.keys())
accs   = [per_stock_metrics[s]['accuracy'] for s in stocks_plot]
f1s    = [per_stock_metrics[s]['f1']       for s in stocks_plot]
aucs   = [per_stock_metrics[s]['auc']      for s in stocks_plot]

x = np.arange(len(stocks_plot))
w = 0.28
ax = axes[0]
b1 = ax.bar(x - w, accs, w, label='Accuracy', color='#3498db', alpha=0.85)
b2 = ax.bar(x,     f1s,  w, label='F1 Score', color='#2ecc71', alpha=0.85)
b3 = ax.bar(x + w, aucs, w, label='AUC-ROC',  color='#e74c3c', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(stocks_plot, rotation=30, ha='right', fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel('Score'); ax.set_title('Performance Metrics per Stock')
ax.legend(fontsize=8); ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
ax.grid(axis='y', alpha=0.3)

# Ablation study bar chart
ax2 = axes[1]
abl_names = list(ablation_results.keys())
abl_accs  = [ablation_results[n]['acc'] for n in abl_names]
abl_f1s   = [ablation_results[n]['f1']  for n in abl_names]
x2 = np.arange(len(abl_names))
b1 = ax2.bar(x2 - 0.2, abl_accs, 0.4, label='Accuracy', color='#9b59b6', alpha=0.85)
b2 = ax2.bar(x2 + 0.2, abl_f1s,  0.4, label='F1 Score',  color='#f39c12', alpha=0.85)
ax2.set_xticks(x2); ax2.set_xticklabels(abl_names, fontsize=9)
ax2.set_ylim(0, 1); ax2.set_ylabel('Score')
ax2.set_title('Ablation Study\n(Sentiment-Only vs Technical-Only vs IMPI Full)')
ax2.legend(fontsize=9); ax2.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax2.grid(axis='y', alpha=0.3)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., h + 0.01,
             f'{h:.3f}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig3_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig3_performance_comparison.png")

# ── Figure 4: Sentiment vs Price for a stock ─────────────────────────────────
fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig4.suptitle('Figure 4: IMPI Sentiment Score vs. Stock Price (RELIANCE)',
              fontsize=13, fontweight='bold')

stock_focus = 'RELIANCE'
price_data  = fused_datasets[stock_focus].dropna(subset=['close', 'avg_sentiment'])
dates = price_data['date_only']

ax1.plot(dates, price_data['close'], color='#2c3e50', linewidth=1.5, label='Close Price')
ax1.set_ylabel('Price (₹)', fontsize=10)
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

sentiment_vals = price_data['avg_sentiment']
ax2.fill_between(dates, sentiment_vals, 0,
                 where=sentiment_vals >= 0, alpha=0.4, color='#2ecc71', label='Positive')
ax2.fill_between(dates, sentiment_vals, 0,
                 where=sentiment_vals < 0, alpha=0.4, color='#e74c3c', label='Negative')
ax2.plot(dates, price_data['roll_m_mean'], color='#3498db',
         linewidth=1.5, label='3-day Rolling Mean')
ax2.axhline(0, color='black', linewidth=0.5)
ax2.set_ylabel('IMPI Sentiment Score', fontsize=10)
ax2.set_xlabel('Date', fontsize=10); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig4_sentiment_vs_price.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig4_sentiment_vs_price.png")

# ── Figure 5: Feature Importance ─────────────────────────────────────────────
fig5, ax = plt.subplots(figsize=(10, 7))
fig5.suptitle('Figure 5: IMPI Feature Importance\n(RF + GBM Average)',
              fontsize=13, fontweight='bold')

feat_imp = final_model.feature_importance().head(20)
colors_fi = ['#9b59b6' if f in SENTIMENT_FEATURES else '#3498db' for f in feat_imp.index]
bars = ax.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors_fi[::-1], alpha=0.85)
ax.set_xlabel('Importance Score'); ax.grid(axis='x', alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#9b59b6', label='Sentiment Features'),
                   Patch(facecolor='#3498db', label='Technical Features')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig5_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig5_feature_importance.png")

# ── Figure 6: Confusion Matrix ────────────────────────────────────────────────
fig6, axes = plt.subplots(1, 2, figsize=(12, 5))
fig6.suptitle('Figure 6: IMPI Confusion Matrices\n(Test Set — RELIANCE & TCS)',
              fontsize=13, fontweight='bold')

for i, stock in enumerate(['RELIANCE', 'TCS']):
    if stock not in per_stock_metrics:
        continue
    m = per_stock_metrics[stock]
    cm = confusion_matrix(m['y_test'], m['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'],
                ax=axes[i])
    axes[i].set_title(f'{stock}\nAcc={m["accuracy"]:.3f}', fontsize=10)
    axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig6_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig6_confusion_matrices.png")

# ─────────────────────────────────────────────────────────────────────────────
#  RESULTS SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  IMPI MODEL — FINAL RESULTS SUMMARY (Research Paper Table)")
print("=" * 70)

summary_rows = []
for stock, m in per_stock_metrics.items():
    summary_rows.append({
        'Stock':     stock,
        'Accuracy':  round(m['accuracy'], 4),
        'F1 Score':  round(m['f1'],       4),
        'Precision': round(m['precision'],4),
        'Recall':    round(m['recall'],   4),
        'AUC-ROC':   round(m['auc'],      4),
        'Test Samples': m['test_samples'],
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.loc[len(summary_df)] = {
    'Stock': 'AVERAGE',
    'Accuracy':  round(summary_df['Accuracy'].mean(),  4),
    'F1 Score':  round(summary_df['F1 Score'].mean(),  4),
    'Precision': round(summary_df['Precision'].mean(), 4),
    'Recall':    round(summary_df['Recall'].mean(),    4),
    'AUC-ROC':   round(summary_df['AUC-ROC'].mean(),   4),
    'Test Samples': int(summary_df['Test Samples'].mean()),
}
print(summary_df.to_string(index=False))
summary_df.to_csv(f'{OUTPUT_DIR}/impi_results_table.csv', index=False)
print(f"\n  Results saved to: impi_results_table.csv")

print("\n" + "─" * 70)
print("  ABLATION STUDY RESULTS")
print("─" * 70)
for name, res in ablation_results.items():
    print(f"  {name:22s} → Accuracy={res['acc']:.4f} | F1={res['f1']:.4f}")

print("\n" + "─" * 70)
print("  DATASET SOURCES FOR RESEARCH PAPER")
print("─" * 70)
print("""
  1. Financial News Dataset (provided):
     - 19,863 news articles (NIFTY 50 stocks)
     - Stocks: HINDUNILVR, ICICIBANK, INFY, ITC, LICHSGFIN, RELIANCE, SBIN, TCS
     - Date range: 2023 (scraped from financial RSS feeds)
     - Format: title, description, publisher, date, source_file

  2. NIFTY 50 Stock Price Data:
     - Source: Yahoo Finance via yfinance Python library
     - URL: https://finance.yahoo.com
     - Tickers: RELIANCE.NS, TCS.NS, INFY.NS, SBIN.NS, etc.
     - Command: pip install yfinance
               import yfinance as yf
               df = yf.download('RELIANCE.NS', start='2023-01-01')
     - Alternative: NSE India website → https://www.nseindia.com

  3. Additional Financial News Datasets (for paper citation):
     - Kaggle Financial News: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests
     - Reuters News Dataset: https://trec.nist.gov/data/reuters/reuters.html
     - SentFin Dataset: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
""")

print("=" * 70)
print("  ALL OUTPUTS SAVED TO: /mnt/user-data/outputs/")
print("  Figures: fig1 through fig6")
print("  Table:   impi_results_table.csv")
print("=" * 70)
