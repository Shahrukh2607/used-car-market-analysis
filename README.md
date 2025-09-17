# 🚗 Unlocking Value: A Data-Driven Analysis of the Used Car Market  

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)  
[![Selenium](https://img.shields.io/badge/Selenium-WebDriver-brightgreen.svg)](https://www.selenium.dev/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

---

## 📌 Project Overview
This project explores resale trends in the Indian used car market by analyzing **Maruti Suzuki cars** across three cities — **Ahmedabad, Mumbai, and Bengaluru**.  
We collected data from [Cars24.com](https://www.cars24.com) using **Python + Selenium**, processed it with **Pandas**, and created visualizations to uncover buyer preferences and market dynamics.  

---

## 🔍 Key Features
- **Data Collection:**  
  - Scraped 780+ listings with Selenium, handling infinite scroll, dynamic content, and bot detection.  
  - Extracted attributes: price, kilometers driven, fuel type, transmission, year, number of owners, and city.  

- **Data Cleaning & Processing:**  
  - Removed duplicates, standardized price/km formats, and fixed missing values.  

- **Analysis & Visualizations:**  
  - 📊 City-wise average resale prices  
  - ⛽ Fuel type distribution  
  - ⚙️ Transmission pricing trends (manual vs. automatic)  
  - 📉 Depreciation curve (price vs. kilometers driven)  

---

## 📈 Insights
- **Bengaluru** had the highest resale values (~₹4.65 lakh), while **Ahmedabad** had the lowest (~₹3.41 lakh).  
- **Automatic cars** carried a significant premium (~₹1.7 lakh higher than manuals).  
- **Petrol cars** dominated (>80%), while CNG presence was negligible in Bengaluru.  

---

## ⚙️ Tech Stack
- **Python** (Pandas, Matplotlib)  
- **Selenium WebDriver**  
- **Jupyter Notebook**  
- **Microsoft Edge + msedgedriver**  

---

## 📂 Repository Structure
.
├── analyze_maruti.py # Data analysis script
├── visualization.py # Chart generation script
├── Cars_Data_With_Multiple_Cities.ipynb # Notebook workflow
├── maruti_cars_multiple_cities.xlsx # Dataset
├── /outputs/ # Generated charts & PDF
│ ├── bar_avg_price_per_city.png
│ ├── hist_transmission_by_city.png
│ ├── pie_fuel_distribution.png
│ ├── scatter_price_vs_km.png
│ └── maruti_visualizations.pdf
├── /Cars_Data_With_Multiple_Cities/analysis/cars24(1).pptx # Final presentation

---

## 🚀 Future Scope
- Extend analysis to more brands and cities  
- Add predictive modeling for resale price forecasting  
- Build an interactive dashboard (Streamlit / Plotly Dash)  

---

## 📜 License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).  
Feel free to use, modify, and share.

---

👤 **Author:** Mohammed Shahrukh  
📧 Reach out for collaboration or feedback!  
