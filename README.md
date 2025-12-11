# auto-valuate-global
AI-powered used car price predictor for UK and Indian markets
## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer
The car dataset used in this project is sourced from open-source contributions on Kaggle and is used here for educational and portfolio purposes only.
##
# 🚘 AutoValuate Global: AI-Powered Car Price Predictor

![Python](https://img.shields.io/badge/Python-3.9-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📌 Project Overview
**AutoValuate Global** is an end-to-end Machine Learning application designed to predict the fair market value of used cars with high precision.

Unlike standard prediction models, this application features a **Dynamic Localization Engine** that allows it to serve multiple international markets (UK & India) from a single core model. It handles real-time unit conversion (Miles $\leftrightarrow$ Km, GBP $\leftrightarrow$ INR) and provides users with actionable "Deal Ratings" to identify undervalued assets.

## 🚀 Key Features
* **🌍 Multi-Market Architecture:** Seamlessly toggles between UK (GBP/Miles) and Indian (INR/Km) markets using a unified backend model.
* **🧠 Intelligent Valuation:** Powered by a Random Forest Regressor trained on **100,000+** cleaned data points.
* **📊 Deal Analytics:** Visualizes where a car sits in the market distribution (e.g., "Great Deal" vs. "Overpriced").
* **🎨 Professional UI:** Features a modern, responsive Streamlit dashboard with "Glassmorphism" design elements and interactive charts.
* **🛠️ Robust ETL Pipeline:** Custom cleaning scripts to standardize messy data from 9 different manufacturers (Audi, BMW, Ford, etc.).

## 🛠️ Tech Stack
* **Core:** Python 3.9
* **Frontend:** Streamlit (Web App Framework)
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

## 📂 Project Structure
```text
auto-valuate-global/
│
├── app.py                # Main Application Code (Streamlit Frontend)
├── train_model.py        # ML Training Script (Generates the AI Model)
├── requirements.txt      # List of dependencies
├── README.md             # Project Documentation
├── LICENSE               # MIT License
└── data/                 # Data Storage
    └── reference_data.csv  # Sample data for market graphs
