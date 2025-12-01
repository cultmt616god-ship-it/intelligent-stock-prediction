# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:36:49 2019

@author: Kaushik
"""
#**************** IMPORT PACKAGES ********************
from flask import Flask, render_template, request, flash, redirect, url_for, session, abort
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
# Replaced Twitter API with free news-based sentiment analysis
# Twitter imports removed - using Finviz + FinVADER instead
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
from news_sentiment import retrieving_news_polarity, finviz_finvader_sentiment
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from decimal import Decimal
from functools import wraps
import secrets

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#***************** FLASK *****************************
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'CHANGE_ME_IN_PRODUCTION')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///jks_management.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    wallet_balance = db.Column(db.Numeric(12, 2), nullable=False, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login_at = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Company(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(16), unique=True, nullable=False)
    name = db.Column(db.String(255))
    exchange = db.Column(db.String(64))
    sector = db.Column(db.String(128))
    is_active = db.Column(db.Boolean, default=True)


class Broker(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255))
    commission_rate = db.Column(db.Numeric(5, 2), nullable=False, default=0)
    is_active = db.Column(db.Boolean, default=True)


class PortfolioItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=0)
    average_buy_price = db.Column(db.Numeric(12, 2), nullable=False, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('portfolio_items', lazy=True))
    company = db.relationship('Company')


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'), nullable=False)
    txn_type = db.Column(db.String(16), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Numeric(12, 2), nullable=False)
    total_amount = db.Column(db.Numeric(12, 2), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    description = db.Column(db.String(255))
    broker_id = db.Column(db.Integer, db.ForeignKey('broker.id'))
    commission_amount = db.Column(db.Numeric(12, 2), nullable=False, default=0)
    user = db.relationship('User', backref=db.backref('transactions', lazy=True))
    company = db.relationship('Company')
    broker = db.relationship('Broker')


class Dividend(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portfolio_item_id = db.Column(db.Integer, db.ForeignKey('portfolio_item.id'), nullable=False)
    amount_per_share = db.Column(db.Numeric(12, 4), nullable=False)
    total_amount = db.Column(db.Numeric(12, 2), nullable=False)
    payable_date = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    portfolio_item = db.relationship('PortfolioItem', backref=db.backref('dividends', lazy=True))


def generate_csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['csrf_token'] = token
    return token


def verify_csrf():
    token = session.get('csrf_token')
    form_token = request.form.get('csrf_token')
    if not token or not form_token or token != form_token:
        abort(400)


app.jinja_env.globals['csrf_token'] = generate_csrf_token


def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            user_id = session.get('user_id')
            user_role = session.get('user_role')
            if not user_id:
                return redirect(url_for('login'))
            if role and user_role != role:
                abort(403)
            return f(*args, **kwargs)
        return wrapped
    return decorator


def get_current_user():
    user_id = session.get('user_id')
    if not user_id:
        return None
    return User.query.get(user_id)


def get_latest_close_price(symbol):
    end = datetime.now()
    start = end - dt.timedelta(days=10)
    data = yf.download(symbol, start=start, end=end)
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return float(data['Close'].iloc[-1])


def get_active_broker():
    return Broker.query.filter_by(is_active=True).order_by(Broker.id.asc()).first()


def calculate_commission(total_amount, broker):
    if not broker:
        return Decimal('0')
    try:
        rate = Decimal(broker.commission_rate) / Decimal('100')
    except Exception:
        return Decimal('0')
    commission = total_amount * rate
    return commission.quantize(Decimal('0.01'))


with app.app_context():
    db.create_all()

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        verify_csrf()
        email = request.form.get('email', '').strip().lower()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        if not email or not username or not password or not confirm_password:
            flash('All fields are required.', 'danger')
            return render_template('register.html', email=email, username=username)
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html', email=email, username=username)
        existing = User.query.filter((User.email == email) | (User.username == username)).first()
        if existing:
            flash('Email or username already registered.', 'danger')
            return render_template('register.html', email=email, username=username)
        password_hash = generate_password_hash(password)
        user = User(email=email, username=username, password_hash=password_hash, role='user')
        db.session.add(user)
        db.session.commit()
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        verify_csrf()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password) or not user.is_active:
            flash('Invalid credentials.', 'danger')
            return render_template('login.html', email=email)
        session.clear()
        session['user_id'] = user.id
        session['user_role'] = user.role
        user.last_login_at = datetime.utcnow()
        db.session.commit()
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required()
def dashboard():
    user = get_current_user()
    items = PortfolioItem.query.filter_by(user_id=user.id).all()
    transactions = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.created_at.desc()).limit(20).all()
    total_invested = Decimal('0')
    total_current = Decimal('0')
    for item in items:
        invested = Decimal(item.average_buy_price) * Decimal(item.quantity)
        total_invested += invested
        last_price = get_latest_close_price(item.company.symbol) or float(item.average_buy_price)
        total_current += Decimal(str(last_price)) * Decimal(item.quantity)
    return render_template('dashboard.html', user=user, items=items, transactions=transactions,
                           total_invested=total_invested, total_current=total_current)


@app.route('/trade/buy', methods=['POST'])
@login_required()
def trade_buy():
    verify_csrf()
    user = get_current_user()
    symbol = request.form.get('symbol', '').strip().upper()
    quantity_raw = request.form.get('quantity', '0').strip()
    try:
        quantity = int(quantity_raw)
    except ValueError:
        flash('Quantity must be an integer.', 'danger')
        return redirect(url_for('dashboard'))
    if quantity <= 0:
        flash('Quantity must be greater than zero.', 'danger')
        return redirect(url_for('dashboard'))
    if not symbol:
        flash('Symbol is required.', 'danger')
        return redirect(url_for('dashboard'))
    price = get_latest_close_price(symbol)
    if price is None:
        flash('Unable to fetch latest price for symbol.', 'danger')
        return redirect(url_for('dashboard'))
    total = Decimal(str(price)) * Decimal(quantity)
    broker = get_active_broker()
    commission = calculate_commission(total, broker)
    if user.wallet_balance < total + commission:
        flash('Insufficient wallet balance including commission.', 'danger')
        return redirect(url_for('dashboard'))
    company = Company.query.filter_by(symbol=symbol).first()
    if not company:
        company = Company(symbol=symbol, name=symbol)
        db.session.add(company)
        db.session.flush()
    item = PortfolioItem.query.filter_by(user_id=user.id, company_id=company.id).first()
    if item:
        current_total = Decimal(item.average_buy_price) * Decimal(item.quantity)
        new_total = current_total + total
        new_quantity = item.quantity + quantity
        item.average_buy_price = new_total / Decimal(new_quantity)
        item.quantity = new_quantity
    else:
        item = PortfolioItem(user_id=user.id, company_id=company.id, quantity=quantity,
                             average_buy_price=total / Decimal(quantity))
        db.session.add(item)
    user.wallet_balance = user.wallet_balance - (total + commission)
    if broker and commission > 0:
        description = f'Simulated buy order via {broker.name} ({broker.commission_rate}% commission)'
    else:
        description = 'Simulated buy order'
    txn = Transaction(user_id=user.id, company_id=company.id, txn_type='BUY', quantity=quantity,
                      price=Decimal(str(price)), total_amount=total,
                      commission_amount=commission, broker_id=broker.id if broker else None,
                      description=description)
    db.session.add(txn)
    db.session.commit()
    flash('Buy order executed in simulated portfolio.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/trade/sell', methods=['POST'])
@login_required()
def trade_sell():
    verify_csrf()
    user = get_current_user()
    symbol = request.form.get('symbol', '').strip().upper()
    quantity_raw = request.form.get('quantity', '0').strip()
    try:
        quantity = int(quantity_raw)
    except ValueError:
        flash('Quantity must be an integer.', 'danger')
        return redirect(url_for('dashboard'))
    if quantity <= 0:
        flash('Quantity must be greater than zero.', 'danger')
        return redirect(url_for('dashboard'))
    if not symbol:
        flash('Symbol is required.', 'danger')
        return redirect(url_for('dashboard'))
    company = Company.query.filter_by(symbol=symbol).first()
    if not company:
        flash('No holdings for this symbol.', 'danger')
        return redirect(url_for('dashboard'))
    item = PortfolioItem.query.filter_by(user_id=user.id, company_id=company.id).first()
    if not item or item.quantity < quantity:
        flash('Not enough shares to sell.', 'danger')
        return redirect(url_for('dashboard'))
    price = get_latest_close_price(symbol)
    if price is None:
        flash('Unable to fetch latest price for symbol.', 'danger')
        return redirect(url_for('dashboard'))
    total = Decimal(str(price)) * Decimal(quantity)
    broker = get_active_broker()
    commission = calculate_commission(total, broker)
    item.quantity = item.quantity - quantity
    if item.quantity == 0:
        db.session.delete(item)
    user.wallet_balance = user.wallet_balance + (total - commission)
    if broker and commission > 0:
        description = f'Simulated sell order via {broker.name} ({broker.commission_rate}% commission)'
    else:
        description = 'Simulated sell order'
    txn = Transaction(user_id=user.id, company_id=company.id, txn_type='SELL', quantity=quantity,
                      price=Decimal(str(price)), total_amount=total,
                      commission_amount=commission, broker_id=broker.id if broker else None,
                      description=description)
    db.session.add(txn)
    db.session.commit()
    flash('Sell order executed in simulated portfolio.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/funds/topup', methods=['POST'])
@login_required()
def funds_topup():
    verify_csrf()
    user = get_current_user()
    amount_raw = request.form.get('amount', '0').strip()
    try:
        amount = Decimal(amount_raw)
    except Exception:
        flash('Invalid amount.', 'danger')
        return redirect(url_for('dashboard'))
    if amount <= 0:
        flash('Amount must be greater than zero.', 'danger')
        return redirect(url_for('dashboard'))
    user.wallet_balance = user.wallet_balance + amount
    db.session.commit()
    flash('Wallet balance updated for simulation.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/dividends/record', methods=['POST'])
@login_required()
def record_dividend():
    verify_csrf()
    user = get_current_user()
    symbol = request.form.get('symbol', '').strip().upper()
    amount_per_share_raw = request.form.get('amount_per_share', '0').strip()
    try:
        amount_per_share = Decimal(amount_per_share_raw)
    except Exception:
        flash('Invalid dividend amount.', 'danger')
        return redirect(url_for('dashboard'))
    if amount_per_share <= 0:
        flash('Dividend amount must be greater than zero.', 'danger')
        return redirect(url_for('dashboard'))
    company = Company.query.filter_by(symbol=symbol).first()
    if not company:
        flash('No holdings for this symbol.', 'danger')
        return redirect(url_for('dashboard'))
    item = PortfolioItem.query.filter_by(user_id=user.id, company_id=company.id).first()
    if not item or item.quantity <= 0:
        flash('No holdings for this symbol.', 'danger')
        return redirect(url_for('dashboard'))
    total_amount = amount_per_share * Decimal(item.quantity)
    dividend = Dividend(portfolio_item_id=item.id, amount_per_share=amount_per_share,
                        total_amount=total_amount)
    user.wallet_balance = user.wallet_balance + total_amount
    txn = Transaction(user_id=user.id, company_id=company.id, txn_type='DIVIDEND', quantity=item.quantity,
                      price=amount_per_share, total_amount=total_amount,
                      commission_amount=Decimal('0'), broker_id=None,
                      description='Dividend payout recorded')
    db.session.add(dividend)
    db.session.add(txn)
    db.session.commit()
    flash('Dividend recorded and wallet credited.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/admin')
@login_required(role='admin')
def admin_dashboard():
    user_count = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    broker_count = Broker.query.count()
    transaction_count = Transaction.query.count()
    company_count = Company.query.count()
    recent_transactions = Transaction.query.order_by(Transaction.created_at.desc()).limit(25).all()
    brokers = Broker.query.order_by(Broker.name.asc()).all()
    companies = Company.query.order_by(Company.symbol.asc()).all()

    all_transactions = Transaction.query.all()
    total_commission = Decimal('0')
    total_volume = 0
    txn_type_counts = {}
    symbol_totals = {}
    for t in all_transactions:
        if t.commission_amount is not None:
            total_commission += Decimal(t.commission_amount)
        if t.txn_type in ('BUY', 'SELL'):
            total_volume += t.quantity
        txn_type_counts[t.txn_type] = txn_type_counts.get(t.txn_type, 0) + 1
        symbol = t.company.symbol if t.company else None
        if symbol:
            data = symbol_totals.setdefault(symbol, {'quantity': 0, 'value': Decimal('0')})
            data['quantity'] += t.quantity
            if t.total_amount is not None:
                data['value'] += Decimal(t.total_amount)

    txn_type_labels = list(txn_type_counts.keys())
    txn_type_values = list(txn_type_counts.values())

    top_symbols_sorted = sorted(symbol_totals.items(), key=lambda kv: kv[1]['value'], reverse=True)[:5]
    top_symbol_labels = [s for s, _ in top_symbols_sorted]
    top_symbol_values = [float(stats['value']) for _, stats in top_symbols_sorted]

    return render_template(
        'admin_dashboard.html',
        user_count=user_count,
        active_users=active_users,
        broker_count=broker_count,
        transaction_count=transaction_count,
        company_count=company_count,
        total_commission=total_commission,
        total_volume=total_volume,
        recent_transactions=recent_transactions,
        brokers=brokers,
        txn_type_labels=txn_type_labels,
        txn_type_values=txn_type_values,
        top_symbol_labels=top_symbol_labels,
        top_symbol_values=top_symbol_values,
        companies=companies,
    )


@app.route('/admin/brokers', methods=['POST'])
@login_required(role='admin')
def admin_add_broker():
    verify_csrf()
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    commission_raw = request.form.get('commission_rate', '0').strip()
    if not name:
        flash('Broker name is required.', 'danger')
        return redirect(url_for('admin_dashboard'))
    try:
        commission = Decimal(commission_raw)
    except Exception:
        flash('Invalid commission rate.', 'danger')
        return redirect(url_for('admin_dashboard'))
    if commission < 0:
        flash('Commission rate cannot be negative.', 'danger')
        return redirect(url_for('admin_dashboard'))
    broker = Broker(name=name, email=email or None, commission_rate=commission)
    db.session.add(broker)
    db.session.commit()
    flash('Broker added.', 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/companies', methods=['POST'])
@login_required(role='admin')
def admin_add_company():
    verify_csrf()
    symbol = request.form.get('symbol', '').strip().upper()
    name = request.form.get('name', '').strip()
    exchange = request.form.get('exchange', '').strip()
    sector = request.form.get('sector', '').strip()
    is_active_raw = request.form.get('is_active')
    if not symbol:
        flash('Company symbol is required.', 'danger')
        return redirect(url_for('admin_dashboard'))
    company = Company.query.filter_by(symbol=symbol).first()
    if not company:
        company = Company(symbol=symbol)
        db.session.add(company)
    company.name = name or symbol
    company.exchange = exchange or None
    company.sector = sector or None
    company.is_active = bool(is_active_raw)
    db.session.commit()
    flash('Company saved.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    nm = request.form['nm']

    #**************** FUNCTIONS TO FETCH DATA ***************************
    def get_historical(quote):
        end = datetime.now()
        start = datetime(end.year-2,end.month,end.day)
        data = yf.download(quote, start=start, end=end)
        
        # Flatten MultiIndex columns if present (fix for recent yfinance update)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Reset index to make Date a column
        data = data.reset_index()
        
        df = pd.DataFrame(data=data)
        df.to_csv(''+quote+'.csv', index=False)
        if(df.empty):
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            #Format df
            #Last 2 yrs rows => 502, in ascending order => ::-1
            data=data.head(503).iloc[::-1]
            data=data.reset_index()
            #Keep Required cols only
            df=pd.DataFrame()
            df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']
            df.to_csv(''+quote+'.csv',index=False)
        return

    #******************** ARIMA SECTION ********************
    def ARIMA_ALGO(df):
        uniqueVals = df["Code"].unique()  
        len(uniqueVals)
        df=df.set_index("Code")
        #for daily basis
        def parser(x):
            return datetime.strptime(x, '%Y-%m-%d')
        def arima_model(train, test):
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(6,1 ,0))
                model_fit = model.fit()
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat)
                obs = test[t]
                history.append(obs)
            return predictions
        for company in uniqueVals[:10]:
            data=(df.loc[company,:]).reset_index()
            data['Price'] = data['Close']
            Quantity_date = data[['Price','Date']]
            Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
            Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
            Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
            Quantity_date = Quantity_date.drop(['Date'],axis =1)
            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(Quantity_date, linestyle=':', color='#1F77B4')
            plt.savefig('static/Trends.png')
            plt.close(fig)
            
            quantity = Quantity_date.values
            size = int(len(quantity) * 0.80)
            train, test = quantity[0:size], quantity[size:len(quantity)]
            #fit in model
            predictions = arima_model(train, test)
            
            #plot graph
            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(test, label='Actual Price', linestyle=':', color='#1F77B4')
            plt.plot(predictions, label='Predicted Price', color='#4B73B1')
            plt.legend(loc=4)
            plt.savefig('static/ARIMA.png')
            plt.close(fig)
            print()
            print("##############################################################################")
            arima_pred=predictions[-2]
            print("Tomorrow's",quote," Closing Price Prediction by ARIMA:",arima_pred)
            #rmse calculation
            error_arima = math.sqrt(mean_squared_error(test, predictions))
            print("ARIMA RMSE:",error_arima)
            print("##############################################################################")
            return arima_pred, error_arima
        
        


    #************* LSTM SECTION **********************

    def LSTM_ALGO(df):
        #Split data into training set and test set
        dataset_train=df.iloc[0:int(0.8*len(df)),:]
        dataset_test=df.iloc[int(0.8*len(df)):,:]
        ############# NOTE #################
        #TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
        # HERE N=7
        ###dataset_train=pd.read_csv('Google_Stock_Price_Train.csv')
        training_set=df.iloc[:,4:5].values# 1:2, to store as numpy array else Series obj will be stored
        #select cols using above manner to select as float64 type, view in var explorer

        #Feature Scaling
        from sklearn.preprocessing import MinMaxScaler
        sc=MinMaxScaler(feature_range=(0,1))#Scaled values btween 0,1
        training_set_scaled=sc.fit_transform(training_set)
        #In scaling, fit_transform for training, transform for test
        
        #Creating data stucture with 7 timesteps and 1 output. 
        #7 timesteps meaning storing trends from 7 days before current day to predict 1 next output
        X_train=[]#memory with 7 days from day i
        y_train=[]#day i
        for i in range(7,len(training_set_scaled)):
            X_train.append(training_set_scaled[i-7:i,0])
            y_train.append(training_set_scaled[i,0])
        #Convert list to numpy arrays
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        X_forecast=np.array(X_train[-1,1:])
        X_forecast=np.append(X_forecast,y_train[-1])
        #Reshaping: Adding 3rd dimension
        X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))#.shape 0=row,1=col
        X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
        #For X_train=np.reshape(no. of rows/samples, timesteps, no. of cols/features)
        
        #Building RNN
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        from keras.layers import LSTM
        
        #Initialise RNN
        regressor=Sequential()
        
        #Add first LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
        #units=no. of neurons in layer
        #input_shape=(timesteps,no. of cols/features)
        #return_seq=True for sending recc memory. For last layer, retrun_seq=False since end of the line
        regressor.add(Dropout(0.1))
        
        #Add 2nd LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
        #Add 3rd LSTM layer
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
        #Add 4th LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        
        #Add o/p layer
        regressor.add(Dense(units=1))
        
        #Compile
        regressor.compile(optimizer='adam',loss='mean_squared_error')
        
        #Training
        regressor.fit(X_train,y_train,epochs=25,batch_size=32 )
        #For lstm, batch_size=power of 2
        
        #Testing
        ###dataset_test=pd.read_csv('Google_Stock_Price_Test.csv')
        real_stock_price=dataset_test.iloc[:,4:5].values
        
        #To predict, we need stock prices of 7 days before the test set
        #So combine train and test set to get the entire data set
        dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0) 
        testing_set=dataset_total[ len(dataset_total) -len(dataset_test) -7: ].values
        testing_set=testing_set.reshape(-1,1)
        #-1=till last row, (-1,1)=>(80,1). otherwise only (80,0)
        
        #Feature scaling
        testing_set=sc.transform(testing_set)
        
        #Create data structure
        X_test=[]
        for i in range(7,len(testing_set)):
            X_test.append(testing_set[i-7:i,0])
            #Convert list to numpy arrays
        X_test=np.array(X_test)
        
        #Reshaping: Adding 3rd dimension
        X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        
        #Testing Prediction
        predicted_stock_price=regressor.predict(X_test)
        
        #Getting original prices back from scaled values
        predicted_stock_price=sc.inverse_transform(predicted_stock_price)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(real_stock_price, label='Actual Price', linestyle=':', color='#1F77B4')  
        plt.plot(predicted_stock_price, label='Predicted Price', color='#4B73B1')
          
        plt.legend(loc=4)
        plt.savefig('static/LSTM.png')
        plt.close(fig)
        
        
        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        
        
        #Forecasting Prediction
        forecasted_stock_price=regressor.predict(X_forecast)
        
        #Getting original prices back from scaled values
        forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
        
        lstm_pred=forecasted_stock_price[0,0]
        print()
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by LSTM: ",lstm_pred)
        print("LSTM RMSE:",error_lstm)
        print("##############################################################################")
        return lstm_pred,error_lstm
    #***************** LINEAR REGRESSION SECTION ******************       
    def LIN_REG_ALGO(df):
        #No of days to be forcasted in future
        forecast_out = int(7)
        #Price after n days
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        #New df with only relevant data
        df_new=df[['Close','Close after n days']]

        #Structure data for train, test & forecast
        #lables of known data, discard last 35 rows
        y =np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
        #all cols of known data except lables, discard last 35 rows
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
        #Unknown, X to be forecasted
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])
        
        #Traning, testing to plot graphs, check accuracy
        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]
        
        # Feature Scaling===Normalization
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        X_to_be_forecasted=sc.transform(X_to_be_forecasted)
        
        #Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        
        #Testing
        y_test_pred=clf.predict(X_test)
        y_test_pred=y_test_pred*(1.04)
        import matplotlib.pyplot as plt2
        fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
        plt2.plot(y_test, label='Actual Price', linestyle=':', color='#1F77B4')
        plt2.plot(y_test_pred, label='Predicted Price', color='#4B73B1')
        
        plt2.legend(loc=4)
        plt2.savefig('static/LR.png')
        plt2.close(fig)
        
        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
        
        
        #Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set=forecast_set*(1.04)
        mean=forecast_set.mean()
        lr_pred=forecast_set[0,0]
        print()
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by Linear Regression: ",lr_pred)
        print("Linear Regression RMSE:",error_lr)
        print("##############################################################################")
        return df, lr_pred, forecast_set, mean, error_lr


    def recommending(df, global_polarity,today_stock,mean):
            count=20 #Num of tweets to be displayed on web page
            #Convert to Textblob format for assigning polarity
            tw2 = tweet.full_text
            tw = tweet.full_text
            #Clean
            tw=p.clean(tw)
            #print("-------------------------------CLEANED TWEET-----------------------------")
            #print(tw)
            #Replace &amp; by &
            tw=re.sub('&amp;','&',tw)
            #Remove :
            tw=re.sub(':','',tw)
            #print("-------------------------------TWEET AFTER REGEX MATCHING-----------------------------")
            #print(tw)
            #Remove Emojis and Hindi Characters
            tw=tw.encode('ascii', 'ignore').decode('ascii')

            #print("-------------------------------TWEET AFTER REMOVING NON ASCII CHARS-----------------------------")
            #print(tw)
            blob = TextBlob(tw)
            polarity = 0 #Polarity of single individual tweet
            for sentence in blob.sentences:
                   
                polarity += sentence.sentiment.polarity
                if polarity>0:
                    pos=pos+1
                if polarity<0:
                    neg=neg+1
                
                global_polarity += sentence.sentiment.polarity
            if count > 0:
                tw_list.append(tw2)
                
            tweet_list.append(Tweet(tw, polarity))


    def recommending(df, global_polarity,today_stock,mean):
        if today_stock.iloc[-1]['Close'] < mean:
            if global_polarity > 0:
                idea="RISE"
                decision="BUY"
                print()
                print("##############################################################################")
                print("According to the ML Predictions and Sentiment Analysis, a",idea,"in",quote,"stock is expected => ",decision)
            elif global_polarity <= 0:
                idea="FALL"
                decision="SELL"
                print()
                print("##############################################################################")
                print("According to the ML Predictions and Sentiment Analysis, a",idea,"in",quote,"stock is expected => ",decision)
        else:
            idea="FALL"
            decision="SELL"
            print()
            print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis, a",idea,"in",quote,"stock is expected => ",decision)
        return idea, decision





    #**************GET DATA ***************************************
    quote=nm
    #Try-except to check if valid stock symbol
    try:
        get_historical(quote)
    except:
        return render_template('index.html',not_found=True)
    else:
    
        #************** PREPROCESSUNG ***********************
        df = pd.read_csv(''+quote+'.csv')
        print("##############################################################################")
        print("Today's",quote,"Stock Data: ")
        today_stock=df.iloc[-1:]
        print(today_stock)
        print("##############################################################################")
        df = df.dropna()
        code_list=[]
        for i in range(0,len(df)):
            code_list.append(quote)
        df2=pd.DataFrame(code_list,columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df=df2


        arima_pred, error_arima=ARIMA_ALGO(df)
        lstm_pred, error_lstm=LSTM_ALGO(df)
        df, lr_pred, forecast_set,mean,error_lr=LIN_REG_ALGO(df)
        
        # Use FREE news-based sentiment analysis instead of Twitter
        print()
        print("##############################################################################")
        print("Fetching news sentiment for", quote, "...")
        print("##############################################################################")
        polarity, sentiment_list, sentiment_pol, pos, neg, neutral = finviz_finvader_sentiment(quote)
        
        idea, decision=recommending(df, polarity,today_stock,mean)
        print()
        print("Forecasted Prices for Next 7 days:")
        print(forecast_set)
        today_stock=today_stock.round(2)
        return render_template('results.html',quote=quote,arima_pred=round(arima_pred,2),lstm_pred=round(lstm_pred,2),
                               lr_pred=round(lr_pred,2),open_s=today_stock['Open'].to_string(index=False),
                               close_s=today_stock['Close'].to_string(index=False),
                               sentiment_list=sentiment_list,sentiment_pol=sentiment_pol,idea=idea,decision=decision,high_s=today_stock['High'].to_string(index=False),
                               low_s=today_stock['Low'].to_string(index=False),vol=today_stock['Volume'].to_string(index=False),
                               forecast_set=forecast_set,error_lr=round(error_lr,2),error_lstm=round(error_lstm,2),error_arima=round(error_arima,2))
if __name__ == '__main__':
   app.run()
   

















