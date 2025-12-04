[![DOI](https://zenodo.org/badge/742607049.svg)](https://zenodo.org/doi/10.5281/zenodo.10498988)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis/blob/master/LICENSE)
[![Code Coverage](https://codecov.io/gh/NCSU-Fall-2022-SE-Project-Team-11/XpensAuditor---Group-11/branch/main/graphs/badge.svg)](https://codecov.io)
![GitHub contributors](https://img.shields.io/badge/Contributors-1-brightgreen)
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://github.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis/edit/master/README.md)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis)
![GitHub issues](https://img.shields.io/github/issues/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis)
![GitHub closed issues](https://img.shields.io/github/issues-closed/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis.svg)](https://img.shields.io/github/repo-size/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis.svg)
[![GitHub last commit](https://img.shields.io/github/last-commit/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis)](https://github.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis/commits/master)
![GitHub language count](https://img.shields.io/github/languages/count/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis)
[![Commit Acitivity](https://img.shields.io/github/commit-activity/m/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis)](https://github.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis)
[![Code Size](https://img.shields.io/github/languages/code-size/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis)](mpp-backend)
![GitHub forks](https://img.shields.io/github/forks/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis?style=social)
![GitHub stars](https://img.shields.io/github/stars/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis?style=social)

# Stock Analysis & Market Sentiment System

A full-stack web application that combines machine learning-based stock price prediction with sentiment analysis of financial news and integrated portfolio management. This system provides a simulated trading environment where users can experiment with stock predictions and portfolio strategies in a risk-free setting.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#project-overview">Project Overview</a></li>
    <li><a href="#key-features">Key Features</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#links">Links</a></li>
  </ol>
</details>

## Project Overview

This project presents a Stock Analysis & Market Sentiment system that integrates customer profiles, stock data, and predictive analytics into one full-stack web application. The system provides simulated buy/sell transactions, fund and dividend tracking, broker commission management, and an admin monitoring dashboard.

On the prediction side, an LSTM-based model (with classical ARIMA and Linear Regression baselines) forecasts future stock trends using historical prices, while a sentiment analysis module aggregates recent financial-news sentiment to give additional context. The backend is implemented in Python using Flask, with a SQLite database and responsive web interfaces, and the prediction outputs are visualised through interactive D3-based charts and summary widgets on the results page.

The proposed system is intended for students, researchers, and amateur investors who wish to explore stock prediction and portfolio management in a safe, simulated environment. It closes gaps in the literature by embedding forecasting models into an end-to-end management workflow with explainable dashboards rather than providing prediction in isolation.

## Key Features

### Portfolio Management
- User registration and authentication with role-based access control
- Virtual wallet and fund tracking for simulated trading
- Stock buy/sell operations with commission calculations
- Dividend recording and tracking
- Portfolio holdings visualization and performance metrics

### Prediction Engine
- LSTM neural network for stock price prediction
- Classical baseline models (ARIMA, Linear Regression)
- Financial news sentiment analysis using multiple sources
- Interactive visualization of prediction accuracy
- Seven-day forward price forecasting

### Administration
- User and company management
- Broker configuration with commission rates
- Transaction monitoring and reporting
- System statistics dashboard

## Built With
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Javascript](https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![nodejs](https://img.shields.io/badge/Node.js-43853D?style=for-the-badge&logo=node.js&logoColor=white)
![html](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![css](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![bootstrap](https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white)
![jquery](https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-red?style=for-the-badge&logo=keras&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-blue?style=for-the-badge&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/Pandas-green?style=for-the-badge&logo=pandas&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis.git
cd Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python main.py
```

5. Access the application at `http://localhost:5000`

### Default Credentials
- **Admin User**: 
  - Username: admin
  - Email: stockpredictorapp@gmail.com
  - Password: Samplepass@123

## Authors

### Current Maintainer
This project is now maintained as an open-source educational tool.

### Original Developer
#### Kaushik Jadhav

If you use this software in your research, please cite it as:

**APA Format:**
Jadhav, K. (2023). Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis (Version 2.0.0) [Computer software]. https://doi.org/10.5281/zenodo.10498988

**BibTeX:**
```
@software{Jadhav_Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis_2023,
author = {Jadhav, Kaushik},
doi = {10.5281/zenodo.10498988},
month = mar,
title = Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis,
url = {https://github.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis},
version = {2.0.0},
year = {2023}
}
```

<ul>
<li>Github: https://github.com/kaushikjadhav01</li>
<li>Medium: https://medium.com/@kaushikjadhav01</li>
<li>LinkedIn: https://www.linkedin.com/in/kaushikjadhav01/</li>
<li>Portfolio: http://kajadhav.me/</li>
<li>LinkedIn: https://www.linkedin.com/in/kajadhav/</li>
<li>Dev.to: https://dev.to/kaushikjadhav01</li>
<li>Codesignal: https://app.codesignal.com/profile/kaushik_j_vtc</li>
<li>Google Scholar: https://scholar.google.com/citations?user=iRYcFi0AAAAJ</li>
<li>Daily.dev: https://app.daily.dev/kaushikjadhav01</li>
<li>Google devs: https://developers.google.com/profile/u/kaushikjadhav01</li>
<li>Stack Overflow: https://stackoverflow.com/users/21890981/kaushik-jadhav</li>
</ul>

## Links
* [Issue tracker](https://github.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis/issues)
* [Source code](https://github.com/kaushikjadhav01/Stock-Market-Prediction-Web-App-using-Machine-Learning-And-Sentiment-Analysis)