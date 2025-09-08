from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import requests
import json
import os
import time
from datetime import datetime, timedelta



app = Flask(__name__)

# Define major stocks at module level so it's always available
MAJOR_STOCKS = {
    'RELIANCE.NS': 'Reliance Industries Ltd.',
    'TCS.NS': 'Tata Consultancy Services Ltd.',
    'HDFCBANK.NS': 'HDFC Bank Ltd.',
    'INFY.NS': 'Infosys Ltd.',
    'HINDUNILVR.NS': 'Hindustan Unilever Ltd.',
    'ICICIBANK.NS': 'ICICI Bank Ltd.',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel Ltd.',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank Ltd.',
    'HCLTECH.NS': 'HCL Technologies Ltd.',
    'WIPRO.NS': 'Wipro Ltd.',
    'ROHLTD.NS': 'Royal Orchid Hotels Ltd.',
    'APOLLO.NS': 'Apollo Microsystems Ltd.',
    'BEL.NS': 'Bharat Electronics Ltd.',
    'BOMDYEING.BO': 'Bombay Dyeing & Mfg Co Ltd.',
    'SYRMA.NS': 'Syrma SGS Technology Ltd.',
    'GMDC.BO': 'Gujarat Mineral Development Corporation Ltd.',
    'AXISBANK.NS': 'Axis Bank Ltd.',
    'ASIANPAINT.NS': 'Asian Paints Ltd.',
    'MARUTI.NS': 'Maruti Suzuki India Ltd.',
    'SUNPHARMA.NS': 'Sun Pharmaceutical Industries Ltd.',
    'TATAMOTORS.NS': 'Tata Motors Ltd.',
    'ULTRACEMCO.NS': 'UltraTech Cement Ltd.',
    'TITAN.NS': 'Titan Company Ltd.',
    'BAJFINANCE.NS': 'Bajaj Finance Ltd.',
    'NESTLEIND.NS': 'Nestle India Ltd.',
    'ADANIENT.NS': 'Adani Enterprises Ltd.',
    'COALINDIA.NS': 'Coal India Ltd.',
    'POWERGRID.NS': 'Power Grid Corporation of India Ltd.',
    'NTPC.NS': 'NTPC Ltd.',
    'ONGC.NS': 'Oil and Natural Gas Corporation Ltd.',
    'TATASTEEL.NS': 'Tata Steel Ltd.',
    'JSWSTEEL.NS': 'JSW Steel Ltd.',
    'HINDALCO.NS': 'Hindalco Industries Ltd.',
    'VEDL.NS': 'Vedanta Ltd.',
    'APOLLOHOSP.NS': 'Apollo Hospitals Enterprise Ltd.',
    'CIPLA.NS': 'Cipla Ltd.',
    # Small Cap and PSU Stocks
    'IREDA.NS': 'Indian Renewable Energy Development Agency Ltd.',
    'SUZLON.NS': 'Suzlon Energy Ltd.',
    'YESBANK.NS': 'Yes Bank Ltd.',
    'IDBI.NS': 'IDBI Bank Ltd.',
    'PNB.NS': 'Punjab National Bank',
    'BANKBARODA.NS': 'Bank of Baroda',
    'IRFC.NS': 'Indian Railway Finance Corporation Ltd.',
    'RAILTEL.NS': 'RailTel Corporation of India Ltd.',
    'RVNL.NS': 'Rail Vikas Nigam Ltd.',
    'HUDCO.NS': 'Housing & Urban Development Corporation Ltd.',
    'MAHABANK.NS': 'Bank of Maharashtra',
    'CANBK.NS': 'Canara Bank',
    'UNIONBANK.NS': 'Union Bank of India',
    'NATIONALUM.NS': 'National Aluminium Company Ltd.',
    'RECLTD.NS': 'REC Ltd.',
    'NHPC.NS': 'NHPC Ltd.',
    'SUPREMEIND.NS': 'Supreme Industries Ltd.',
    'COCHINSHIP.NS': 'Cochin Shipyard Ltd.',
    'MIDHANI.NS': 'Mishra Dhatu Nigam Ltd.',
    'IEX.NS': 'Indian Energy Exchange Ltd.',
    'PERSISTENT.NS': 'Persistent Systems Ltd.',
    'LICI.NS': 'Life Insurance Corporation of India',
    'ZOMATO.NS': 'Zomato Ltd.',
    'PAYTM.NS': 'One 97 Communications Ltd.',
    'NYKAA.NS': 'FSN E-Commerce Ventures Ltd.',
    'BRITANNIA.NS': 'Britannia Industries Ltd.',
    'EICHERMOT.NS': 'Eicher Motors Ltd.',
    'HEROMOTOCO.NS': 'Hero MotoCorp Ltd.',
    'BAJAJ-AUTO.NS': 'Bajaj Auto Ltd.',
    'TECHM.NS': 'Tech Mahindra Ltd.',
    'LT.NS': 'Larsen & Toubro Ltd.',
    'IRCTC.NS': 'Indian Railway Catering and Tourism Corporation Ltd.',
    'ITC.NS': 'ITC Ltd.',
    'INDUSINDBK.NS': 'IndusInd Bank Ltd.',
    'ADANIPORTS.NS': 'Adani Ports and Special Economic Zone Ltd.',
    'ADANIGREEN.NS': 'Adani Green Energy Ltd.',
    'ADANIPOWER.NS': 'Adani Power Ltd.',
    'PIDILITIND.NS': 'Pidilite Industries Ltd.',
    'DABUR.NS': 'Dabur India Ltd.',
    'ZOMATO.NS': 'Zomato Ltd.',
    'NYKAA.NS': 'FSN E-Commerce Ventures Ltd.',
    'PAYTM.NS': 'One 97 Communications Ltd.',
    'RELIANCE.NS': 'Reliance Industries Ltd.',
    'TCS.NS': 'Tata Consultancy Services Ltd.',
    'HDFCBANK.NS': 'HDFC Bank Ltd.',
    'INFY.NS': 'Infosys Ltd.',
    'HINDUNILVR.NS': 'Hindustan Unilever Ltd.',
    'ICICIBANK.NS': 'ICICI Bank Ltd.',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel Ltd.',
    'KOTAKBANK.NS': 'Kotak Mahindra Bank Ltd.',
    'HCLTECH.NS': 'HCL Technologies Ltd.',
    'WIPRO.NS': 'Wipro Ltd.',
    'AXISBANK.NS': 'Axis Bank Ltd.',
    'ASIANPAINT.NS': 'Asian Paints Ltd.',
    'MARUTI.NS': 'Maruti Suzuki India Ltd.',
    'SUNPHARMA.NS': 'Sun Pharmaceutical Industries Ltd.',
    'TATAMOTORS.NS': 'Tata Motors Ltd.',
    'ULTRACEMCO.NS': 'UltraTech Cement Ltd.',
    'TITAN.NS': 'Titan Company Ltd.',
    'BAJFINANCE.NS': 'Bajaj Finance Ltd.',
    'NESTLEIND.NS': 'Nestle India Ltd.',
    'ADANIENT.NS': 'Adani Enterprises Ltd.',
    'COALINDIA.NS': 'Coal India Ltd.',
    'POWERGRID.NS': 'Power Grid Corporation of India Ltd.',
    'NTPC.NS': 'NTPC Ltd.',
    'ONGC.NS': 'Oil and Natural Gas Corporation Ltd.',
    'TATASTEEL.NS': 'Tata Steel Ltd.',
    'JSWSTEEL.NS': 'JSW Steel Ltd.',
    'HINDALCO.NS': 'Hindalco Industries Ltd.',
    'VEDL.NS': 'Vedanta Ltd.',
    'APOLLOHOSP.NS': 'Apollo Hospitals Enterprise Ltd.',
    'CIPLA.NS': 'Cipla Ltd.',
    'DIVISLAB.NS': 'Divi\'s Laboratories Ltd.',
    'DRREDDY.NS': 'Dr. Reddy\'s Laboratories Ltd.',
    'BRITANNIA.NS': 'Britannia Industries Ltd.',
    'EICHERMOT.NS': 'Eicher Motors Ltd.',
    'HEROMOTOCO.NS': 'Hero MotoCorp Ltd.',
    'BAJAJ-AUTO.NS': 'Bajaj Auto Ltd.',
    'TECHM.NS': 'Tech Mahindra Ltd.',
    'LT.NS': 'Larsen & Toubro Ltd.',
    'IRCTC.NS': 'Indian Railway Catering and Tourism Corporation Ltd.',
    'ITC.NS': 'ITC Ltd.',
    'INDUSINDBK.NS': 'IndusInd Bank Ltd.',
    'ADANIPORTS.NS': 'Adani Ports and Special Economic Zone Ltd.',
    'ADANIGREEN.NS': 'Adani Green Energy Ltd.',
    'ADANIPOWER.NS': 'Adani Power Ltd.',
    'PIDILITIND.NS': 'Pidilite Industries Ltd.',
    'DABUR.NS': 'Dabur India Ltd.',
    'ZOMATO.NS': 'Zomato Ltd.',
    'NYKAA.NS': 'FSN E-Commerce Ventures Ltd.',
    'PAYTM.NS': 'One 97 Communications Ltd.',
    'IREDA.NS': 'Indian Renewable Energy Development Agency Ltd.',
    'CENTRAL.NS': 'Central Bank of India',
    'UNIONBANK.NS': 'Union Bank of India',
    'SUZLON.NS': 'Suzlon Energy Ltd.',
    'YESBANK.NS': 'Yes Bank Ltd.',
    'IDBI.NS': 'IDBI Bank Ltd.',
    'PNB.NS': 'Punjab National Bank',
    'BANKBARODA.NS': 'Bank of Baroda',
    'CANBK.NS': 'Canara Bank',
    'IRFC.NS': 'Indian Railway Finance Corporation Ltd.',
    'RAILTEL.NS': 'RailTel Corporation of India Ltd.',
    'HUDCO.NS': 'Housing & Urban Development Corporation Ltd.',
    'RVNL.NS': 'Rail Vikas Nigam Ltd.',
    'MAHABANK.NS': 'Bank of Maharashtra',
    'NATIONALUM.NS': 'National Aluminium Company Ltd.',
    'RECLTD.NS': 'REC Ltd.',
    'NHPC.NS': 'NHPC Ltd.',
    'SUPREMEIND.NS': 'Supreme Industries Ltd.',
    'COCHINSHIP.NS': 'Cochin Shipyard Ltd.',
    'MIDHANI.NS': 'Mishra Dhatu Nigam Ltd.'
}

def fetch_nse_stocks():
    cache_file = 'stock_cache.json'
    cache_expiry = 24  # hours
    
    # Check if cache exists and is recent
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
            last_updated = datetime.fromisoformat(cache_data['last_updated'])
            if datetime.now() - last_updated < timedelta(hours=cache_expiry):
                return cache_data['stocks']

    # List of major Indian indices
    indices = [
        "^NSEI",  # NIFTY 50
        "^NSEBANK",  # NIFTY BANK
        "^CNXIT",  # NIFTY IT
        "^CNXAUTO",  # NIFTY AUTO
        "^CNXFMCG",  # NIFTY FMCG
        "^CNXPHARMA",  # NIFTY PHARMA
        "NIFTYMIDCAP.NS",  # NIFTY MIDCAP 100
        "NIFTYSMALLCAP.NS"  # NIFTY SMALLCAP 100
    ]
    
    all_stocks = {}
    
    try:
        urls = [
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=IN&scrIds=all_stocks_in_in&start=0&count=500",
            "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=IN&scrIds=small_cap_stocks_in_in&start=0&count=500"
        ]
        
        for url in urls:
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json'
            }
            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()
            
            if 'finance' in data and 'result' in data['finance'] and data['finance']['result']:
                quotes = data['finance']['result'][0].get('quotes', [])
                for quote in quotes:
                    if quote.get('symbol', '').endswith('.NS'):
                        all_stocks[quote['symbol']] = quote.get('shortName', quote['symbol'].replace('.NS', ''))

        # Add our curated list of major stocks
        all_stocks.update(MAJOR_STOCKS)
        
        # Cache the results
        cache_data = {
            'last_updated': datetime.now().isoformat(),
            'stocks': all_stocks
        }
        
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        cache_file = os.path.join(cache_dir, 'stock_cache.json')
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            print(f"Warning: Could not write cache file: {str(e)}")
            
        return all_stocks
    except Exception as e:
        print(f"Error fetching stocks: {str(e)}")
        return MAJOR_STOCKS
        # Fetch top companies by market cap from Yahoo Finance
         # Fall back to major stocks list if there's any error
        
    #     all_stocks.update(major_stocks)
        
    #     # Cache the results
    #     cache_data = {
    #         'last_updated': datetime.now().isoformat(),
    #         'stocks': all_stocks
    #     }
    #     with open(cache_file, 'w') as f:
    #         json.dump(cache_data, f)
            
    #     return all_stocks
    # except Exception as e:
    #     print(f"Error fetching stocks: {str(e)}")
    #     # Fallback to major stocks if there's an error
    #     return major_stocks

# Initialize the stock dictionary
INDIAN_STOCKS = fetch_nse_stocks()
print("INDIAN_STOCKS:")  # Debugging line to see the fetched stocks
print(INDIAN_STOCKS)  # Debugging line to see the fetched stocks

# @app.route('/suggest', methods=['GET'])
# def suggest_stocks():
#     query = request.args.get('query', '').upper()
#     if not query:
#         return jsonify([])
        
#     suggestions = []
#     for symbol, name in INDIAN_STOCKS.items():
#         if (query in symbol.upper().replace('.NS', '') or 
#             query in name.upper()):
#             suggestions.append({
#                 'symbol': symbol.replace('.NS', ''),
#                 'name': name
#             })
    
#     return jsonify(suggestions[:10])  # Limit to 10 suggestions

def fetch_stock_data(symbol):
    print(f"\nFetching data for symbol: {symbol}")
    
    # Alpha Vantage API key - Replace with your API key
    API_KEY = "1GRAK4YF5O10LHE1"  # New API key with higher rate limit
    
    def get_alpha_vantage_data(symbol_suffix):
        try:
            base_url = "https://www.alphavantage.co/query"
            # Start directly with TIME_SERIES_DAILY instead of GLOBAL_QUOTE
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol_suffix,
                "outputsize": "full",
                "apikey": API_KEY,
                "market": "CNX"  # Adding back CNX market for Indian stocks
            }
            
            print(f"Making request to Alpha Vantage for symbol: {symbol_suffix}")
            print(f"Full URL: {base_url}?{'&'.join(f'{k}={v}' for k, v in params.items())}")
            
            response = requests.get(base_url, params=params)
            print(f"Response status code: {response.status_code}")
            
            data = response.json()
            print(f"API Response keys: {list(data.keys())}")  # Print only the keys to avoid huge output
            if "Information" in data.keys():
                print(f"Alpha Vantage Information: {data['Information']}")

            # Debug the API response
            if "Error Message" in data:
                print(f"Alpha Vantage Error: {data['Error Message']}")
                return pd.DataFrame()
            
            if "Note" in data:
                print(f"Alpha Vantage Note: {data['Note']}")
                # If we hit API limit, wait for a moment and try again
                if "API call frequency" in data["Note"]:
                    print("Waiting for 60 seconds due to API limit...")
                    time.sleep(60)
                    return get_alpha_vantage_data(symbol_suffix)
            
            if "Time Series (Daily)" in data:
                daily_data = data["Time Series (Daily)"]
                if not daily_data:
                    print(f"No daily data found for {symbol_suffix}")
                    return pd.DataFrame()
                    
                df = pd.DataFrame.from_dict(daily_data, orient='index')
                
                # Rename columns to match previous format
                df = df.rename(columns={
                    '1. open': 'Open',
                    '2. high': 'High',
                    '3. low': 'Low',
                    '4. close': 'Close',
                    '5. volume': 'Volume'
                })
                
                # Convert string values to float
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Sort index to ascending order
                df = df.sort_index()
                
                # Keep only last year's data
                df.index = pd.to_datetime(df.index)
                one_year_ago = pd.Timestamp.now() - pd.DateOffset(years=1)
                df = df[df.index >= one_year_ago]
                
                if not df.empty:
                    return df
                    
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching {symbol_suffix}: {str(e)}")
            return pd.DataFrame()
    
    # First, strip any existing suffixes and get base symbol
    base_symbol = symbol.replace('.NS', '').replace('.BSE', '').replace('.BO', '')
    
    # For Indian stocks, try BSE format
    suffixes = [
        f"{base_symbol}.BSE"    # Try BSE format      # Fallback to base symbol
    ]
    print(f"Will try the following symbol formats: {suffixes}")
    
    for suffix in suffixes:
        print(f"Trying symbol format: {suffix}")
        data = get_alpha_vantage_data(suffix)
        if not data.empty:
            print(f"Successfully fetched data for {suffix}")
            return data
    
    return pd.DataFrame()  # Return empty DataFrame if no data found

def prepare_data(data):
    print("Starting data preparation...")
    df = data.copy()
    
    # Print initial data info
    print(f"Initial data shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Ensure the DataFrame is sorted by date ascending
    df = df.sort_index(ascending=True)
    print(f"Data after sorting, shape: {df.shape}")
    
    # Helper function to handle division by zero and infinity
    def safe_divide(a, b):
        return np.where(b != 0, a / b, 0)
    
    # Basic price features
    df['Close_Norm'] = safe_divide(df['Close'], df['Close'].iloc[0])  # Normalized closing price
    df['Open_Close_Ratio'] = safe_divide(df['Open'], df['Close'])
    df['High_Low_Ratio'] = safe_divide(df['High'], df['Low'])
    df['True_Range'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1).fillna(df['High'])),
            abs(df['Low'] - df['Close'].shift(1).fillna(df['Low']))
        )
    )

    # Volume features
    df['Volume'] = df['Volume'].fillna(0)  # Handle missing volume data
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean().fillna(df['Volume'])
    df['Volume_MA20'] = df['Volume'].rolling(window=20).mean().fillna(df['Volume'])
    df['Volume_Ratio'] = safe_divide(df['Volume'], df['Volume_MA20'])
    df['Volume_ROC'] = df['Volume'].pct_change(periods=1).fillna(0)

    # Technical indicators
    # Moving averages
    for window in [5, 10, 20, 50, 100]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA{window}_Ratio'] = df['Close'] / df[f'MA{window}']

    # Price momentum
    for period in [5, 10, 20, 50]:
        df[f'ROC{period}'] = df['Close'].pct_change(periods=period)
        df[f'Price_Mom{period}'] = df['Close'] - df['Close'].shift(period)

    # Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    for window in [5, 10, 20]:
        df[f'Volatility{window}'] = df['Daily_Return'].rolling(window=window).std()
        df[f'ATR{window}'] = df['True_Range'].rolling(window=window).mean()

    # Trend indicators
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    # RSI and Stochastic
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = safe_divide(avg_gain, avg_loss)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].clip(0, 100)  # Ensure RSI is between 0 and 100

    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    high_low_diff = high_14 - low_14
    df['K_Percent'] = 100 * safe_divide((df['Close'] - low_14), high_low_diff)
    df['K_Percent'] = df['K_Percent'].clip(0, 100)  # Ensure K% is between 0 and 100
    df['D_Percent'] = df['K_Percent'].rolling(window=3).mean()

    # Bollinger Bands
    for window in [20]:
        ma = df['Close'].rolling(window=window).mean()
        std = df['Close'].rolling(window=window).std()
        df[f'BB_Upper{window}'] = ma + (2 * std)
        df[f'BB_Lower{window}'] = ma - (2 * std)
        df[f'BB_Width{window}'] = safe_divide((df[f'BB_Upper{window}'] - df[f'BB_Lower{window}']), ma)

    # Create target variables for different prediction horizons
    for i in range(1, 31):
        df[f'Target_{i}'] = safe_divide(df['Close'].shift(-i), df['Close']) - 1

    # Drop rows with NaN values
    df = df.dropna()
    
    # Replace infinite values with 0
    df = df.replace([np.inf, -np.inf], 0)
    
    # Clip extreme values to reasonable ranges
    for col in df.columns:
        if col != 'Volume' and df[col].dtype in [np.float64, np.float32]:
            q1 = df[col].quantile(0.01)
            q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(q1, q3)

    return df

def train_model(data):
    print("\nStarting model training...")
    models = {}
    
    # Define all potential features
    all_features = [col for col in data.columns if not col.startswith('Target_')]
    print(f"Total features available: {len(all_features)}")
    
    # Remove any problematic features
    exclude_features = ['Dividends', 'Stock Splits']
    all_features = [f for f in all_features if f not in exclude_features]
    print(f"Features after filtering: {len(all_features)}")
    print("Features:", all_features)
    
    # Initialize scaler
    scaler = MinMaxScaler()
    X = data[all_features].copy()
    
    # Print any columns with all zeros or NaN
    zero_cols = X.columns[X.eq(0).all()].tolist()
    nan_cols = X.columns[X.isna().all()].tolist()
    if zero_cols:
        print("Columns with all zeros:", zero_cols)
    if nan_cols:
        print("Columns with all NaN:", nan_cols)
    
    # Replace any infinite values with 0 before scaling
    X = X.replace([np.inf, -np.inf], 0)
    
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=all_features, index=data.index)
    
    # Train a model for each prediction horizon
    for i in range(1, 31):
        y = data[f'Target_{i}']
        
        # Split the data with time-based split
        split_idx = int(len(data) * 0.8)
        X_train = X_scaled_df.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X_scaled_df.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        
        # Create base model for feature importance
        base_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        base_model.fit(X_train, y_train)
        
        # Get feature importance
        importances = pd.Series(base_model.feature_importances_, index=all_features)
        top_features = importances.nlargest(30).index.tolist()
        
        # Train final model with selected features
        X_train_selected = X_train[top_features]
        X_val_selected = X_val[top_features]
        
        # Create and train the final model with optimized parameters
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            warm_start=True  # Enable warm start for iterative training
        )
        
        # Iterative training with early stopping
        best_val_score = float('-inf')
        patience = 5
        patience_counter = 0
        
        for n in range(1, 6):  # Up to 5 iterations
            model.n_estimators = n * 100
            model.fit(X_train_selected, y_train)
            val_score = model.score(X_val_selected, y_val)
            
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        # Validate predictions are reasonable
        val_pred = model.predict(X_val_selected)
        if abs(np.mean(val_pred)) > 0.5:  # If mean prediction is >50% change
            print(f"Warning: Large average prediction for day {i}: {np.mean(val_pred):.2%}")
            continue
        
        # Store model info
        models[i] = {
            'model': model,
            'scaler': scaler,
            'features': top_features,
            'val_score': best_val_score
        }
    
    return models

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/suggest', methods=['GET'])
def suggest_stocks():
    query = request.args.get('query', '').upper()
    if not query:
        return jsonify([])
        
    suggestions = []
    for symbol, name in MAJOR_STOCKS.items():
        symbol_without_ns = symbol.replace('.NS', '')
        if (query in symbol_without_ns.upper() or 
            query in name.upper()):
            # Give higher priority to matches at the start of the symbol or name
            score = 1 if symbol_without_ns.upper().startswith(query) else (
                   2 if name.upper().startswith(query) else 3)
            suggestions.append({
                'symbol': symbol_without_ns,
                'name': name,
                'score': score
            })
    
    # Sort first by score, then by symbol length
    suggestions.sort(key=lambda x: (x['score'], len(x['symbol'])))
    # Remove the score from the response
    suggestions = [{'symbol': s['symbol'], 'name': s['name']} for s in suggestions]
    return jsonify(suggestions[:10])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symbol = request.json['symbol']
        print(f"\nProcessing prediction request for symbol: {symbol}")
        
        data = fetch_stock_data(symbol)
        print(f"Fetched data shape: {data.shape}")
        
        if data.empty:
            return jsonify({'error': 'No data found for the given stock symbol'})
        
        prepared_data = prepare_data(data)
        print(f"Prepared data shape: {prepared_data.shape}")
        
        if prepared_data.empty:
            return jsonify({'error': 'Could not prepare data for prediction'})
            
        models = train_model(prepared_data)
        print(f"Number of trained models: {len(models)}")
        
        if not models:
            return jsonify({'error': 'Could not train prediction models'})
        
        # Make predictions for next 30 days using the new feature set and scaling
        current_price = data['Close'].iloc[-1]
        print(f"Current price: {current_price}")
        
        last_row = prepared_data.iloc[[-1]]
        print("Available features for prediction:", last_row.columns.tolist())
        
        predictions = []
        
        for day in range(1, 31):
            if day not in models:
                print(f"No model available for day {day}")
                continue
                
            model_info = models[day]
            scaler = model_info['scaler']
            features = model_info['features']
            
            print(f"\nPredicting day {day}")
            print(f"Number of features used: {len(features)}")
            
            # Ensure we have all required features
            missing_features = [f for f in features if f not in last_row.columns]
            if missing_features:
                print(f"Warning: Missing features for day {day}: {missing_features}")
                continue
            
            # Get only the features used during training
            X_last = last_row[features]
            
            try:
                # Check for NaN or infinite values
                if X_last.isna().any().any() or np.isinf(X_last.values).any():
                    print(f"Warning: NaN or infinite values found in features for day {day}")
                    X_last = X_last.replace([np.inf, -np.inf], 0)
                    X_last = X_last.fillna(0)
                
                # Scale the features
                X_last_scaled = scaler.transform(X_last)
                
                # Make prediction
                pct_change_pred = model_info['model'].predict(X_last_scaled)[0]
                print(f"Raw prediction: {pct_change_pred}")
                
                # Clip prediction to reasonable range (-50% to +50%)
                pct_change_pred = np.clip(pct_change_pred, -0.5, 0.5)
                print(f"Clipped prediction: {pct_change_pred}")
                
                predicted_price = current_price * (1 + pct_change_pred)
                predictions.append({
                    'day': day,
                    'price': round(predicted_price, 2),
                    'percent_change': round(pct_change_pred * 100, 2)
                })
                print(f"Successfully predicted day {day}")
            except Exception as e:
                print(f"Warning: Main prediction failed for day {day}: {str(e)}")
                print("Error details:", str(e.__class__.__name__))
                
                # Fallback to a simpler prediction based on recent trend
                try:
                    recent_changes = prepared_data['Daily_Return'].tail(5).mean()
                    pct_change_pred = recent_changes * (1 - (day * 0.1))  # Decay factor
                    pct_change_pred = np.clip(pct_change_pred, -0.5, 0.5)
                    predicted_price = current_price * (1 + pct_change_pred)
                    predictions.append({
                        'day': day,
                        'price': round(predicted_price, 2),
                        'percent_change': round(pct_change_pred * 100, 2)
                    })
                    print(f"Successfully used fallback prediction for day {day}")
                except Exception as e2:
                    print(f"Fallback prediction also failed: {str(e2)}")
                    continue
                
        if not predictions:
            return jsonify({'error': 'Could not generate valid predictions'})
            
        return jsonify({
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Only use debug mode when running directly (not through Gunicorn)
    app.run(debug=True)
else:
    # Configure logging for production
    import logging
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
