from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import requests
import json
import os
from datetime import datetime, timedelta

app = Flask(__name__)

MAJOR_STOCKS = {'RELIANCE.NS': 'Reliance Industries Ltd.', 'TCS.NS': 'Tata Consultancy Services Ltd.', 'HDFCBANK.NS': 'HDFC Bank Ltd.', 'INFY.NS': 'Infosys Ltd.', 'HINDUNILVR.NS': 'Hindustan Unilever Ltd.', 'ICICIBANK.NS': 'ICICI Bank Ltd.', 'SBIN.NS': 'State Bank of India', 'BHARTIARTL.NS': 'Bharti Airtel Ltd.', 'KOTAKBANK.NS': 'Kotak Mahindra Bank Ltd.', 'HCLTECH.NS': 'HCL Technologies Ltd.', 'WIPRO.NS': 'Wipro Ltd.', 'ITC.NS': 'ITC Ltd.', 'TITAN.NS': 'Titan Company Ltd.', 'MARUTI.NS': 'Maruti Suzuki India Ltd.', 'ZOMATO.NS': 'Zomato Ltd.', 'NYKAA.NS': 'FSN E-Commerce Ventures Ltd.', 'PAYTM.NS': 'One 97 Communications Ltd.', 'TATAMOTORS.NS': 'Tata Motors Ltd.'}
