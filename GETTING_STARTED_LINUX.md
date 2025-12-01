# Getting Started with Stock Market Prediction App on Linux

This guide provides minimal steps to get the Stock Market Prediction Web App running on a Linux environment.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Installation Steps

### 1. Clone or Download the Repository

```bash
git clone <repository-url>
cd Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis
```

Or download and extract the source code to a directory.

### 2. Set Up Python Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

Note: If you encounter issues with specific packages, you may need to install system dependencies:
```bash
# For Ubuntu/Debian systems
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential
sudo apt-get install libxml2-dev libxslt1-dev zlib1g-dev libffi-dev libssl-dev
```

### 4. Download NLTK Data (Automatic)

The application will automatically download required NLTK data on first run:
- punkt tokenizer
- vader_lexicon for sentiment analysis

### 5. Run the Application

```bash
python main.py
```

The application will start on `http://localhost:5000` by default.

### 6. Access the Web Interface

Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Enter a stock ticker symbol (e.g., AAPL, GOOGL, NVDA)
2. Click "Predict" to analyze the stock
3. View the predictions from ARIMA, LSTM, and Linear Regression models
4. See sentiment analysis from Finviz news headlines

## Notes

- The application uses yfinance to fetch real-time stock data
- Sentiment analysis is performed using Finviz news scraping + VADER sentiment analysis
- Generated plots are saved in the `static/` directory
- CSV files with stock data are created in the root directory

## Troubleshooting

### Common Issues:

1. **Port already in use**: 
   ```bash
   export FLASK_RUN_PORT=5001
   python main.py
   ```

2. **Permission errors**: 
   Make sure you have write permissions in the application directory for saving plots and CSV files.

3. **Missing system dependencies**: 
   Install required system packages as shown in step 3.

### For Production Deployment:

Use Gunicorn as specified in requirements.txt:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 main:app
```

## System Requirements

- Minimum: 4GB RAM, 2GB free disk space
- Recommended: 8GB RAM, 5GB free disk space