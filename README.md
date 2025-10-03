# Time_Series_Project

**Description:**  
This project focuses on forecasting daily energy prices using historical demand, generation, and weather data. Multiple time series models (ARMA, ARIMA, SARIMA, Box-Jenkins) were implemented and evaluated to improve forecasting accuracy and support strategic decision-making in the energy sector.

---

## Project Structure

```text
Time_Series_Project/
|-- Data/
|   |-- final.csv           # Cleaned dataset for modeling
|-- src/
|   |-- app.py              # Streamlit/FastAPI app (optional)
|   |-- clean.py            # Data cleaning and preprocessing scripts
|   |-- main.py             # Run forecasting models
|   |-- proposal.py         # Project proposal / planning
|   |-- toolbox.py          # Helper functions and utilities
|-- requirements.txt        # Project dependencies
|-- Bagepalli_Report.pdf     # Final project report
```
## Installation

Clone the repository:

```bash
git clone https://github.com/apoorvareddy612/Time_Series_Project.git
cd Time_Series_Project
```

Install dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage
Data Preprocessing
```bash
python src/clean.py
```
Run Forecasting Models
```bash
python src/main.py
```
Launch Dash App
```bash
python src/app.py
```

## Key Features

- Forecasting daily energy prices using **ARMA, ARIMA, SARIMA, Box-Jenkins** models  
- Data cleaning and preprocessing pipelines  
- Visualizations of **trends, seasonality, and model performance**  
- PDF report summarizing methodology and results  

---

## Skills Used

- **Programming & Libraries:** Python, Pandas, NumPy, Matplotlib, Seaborn, Statsmodels, Scikit-learn  
- **Techniques & Analysis:** Time Series Analysis, Forecasting, Data Visualization  
- **Deployment & Tools:** Dash  

---

## Optional Notes

- The `toolbox.py` module contains custom utility functions for data preprocessing, statistical analysis, and visualization.  
- For reproducibility, ensure your working directory structure matches the project structure described above.


