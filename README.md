# 🔄 Fuel Sales Time Series Forecasting System

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.x-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-blue.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-lightgreen.svg)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-yellowgreen.svg)](https://catboost.ai/)
[![Prophet](https://img.shields.io/badge/Facebook%20Prophet-v1.1.2-blueviolet)](https://facebook.github.io/prophet/)
[![statsmodels](https://img.shields.io/badge/statsmodels-v0.14.0-informational)](https://www.statsmodels.org/)

📌 **Note:** The full notebook exceeds GitHub's preview limit and cannot be displayed directly.   
> ▶️ [Open in Google Colab](https://colab.research.google.com/drive/1HEmhth4B5h24ksvGZOyfAyPBYyDbfTJX?usp=sharing) to view and execute.

## 📋 Project Overview

This advanced time series forecasting system predicts future fuel sales across multiple retail locations using ensemble machine learning techniques. The system integrates site metadata with historical sales data, incorporates external factors like weather conditions, and employs sophisticated feature engineering to create highly accurate sales forecasts.

The architecture follows a modular pipeline approach, from data acquisition and cleaning through feature engineering to model training and forecast generation. Multiple algorithms are supported with cross-validation capabilities to ensure comprehensive performance evaluation.

## 💼 Business Context

Fuel sales forecasting provides significant business value for retail fuel operations:

- **Revenue Optimization**: Accurately predicting sales enables optimal inventory management and pricing strategies
- **Cost Reduction**: Precise forecasting reduces carrying costs and minimizes stockouts
- **Operational Efficiency**: Time series forecasting enables efficient resource allocation and staff scheduling
- **Strategic Planning**: Long-term forecasts support expansion planning and capital investment decisions

## 📊 Dataset Details

The system works with two primary datasets:

### Fuel Sales Dataset

| Column Name | Description |
|-------------|-------------|
| Day of Month | The day of the month when the fuel sale was recorded. |
| Fuel Grade | The type or grade of fuel sold (e.g., regular, premium, diesel). |
| Measure Names | The name of the measure (e.g., sales volume, revenue). |
| Month | The month when the fuel sale was recorded. |
| Site | The identifier for the site where the fuel sale occurred. |
| Year | The year when the fuel sale was recorded. |
| Measure Values | The numeric value corresponding to the measure (e.g., volume sold). |

### Sites Dataset

| Column Name | Description |
|-------------|-------------|
| GDSO | Site number or identifier. |
| Fuel Brand ID | Identifier for the fuel brand sold at the site. |
| Site Status | Current operational status of the site (e.g., open, closed). |
| LAND / COT | Class of trade, indicating the business type or category. |
| Division | Business division or segment to which the site belongs. |
| Location Type | Type of location (e.g., urban, rural, highway). |
| Site Rank ID | Rank ID for sites, used for categorization or prioritization (non-sequential). |
| City ID | Identifier for the city where the site is located. |
| ST | State where the site is located. |
| County ID | Identifier for the county where the site is located. |
| Store Brand ID | Brand identifier for the store associated with the site. |
| C Store Brand ID | Brand identifier for the convenience store at the site. |
| Number of MPD's | Number of fuel dispensers or multi-product dispensers at the site. |
| Diesel? | Boolean indicating whether diesel fuel is available at the site. |
| Separate Diesel Canopy? | Indicates if there is a separate canopy for diesel fueling. |
| High-Speed Diesel | Indicates the presence of high-speed diesel pumps for trucks. |
| Multi Dealers | Indicates if multiple dealers are present at the site. |
| Auto Service Bay | Indicates the presence of an automotive service bay at the site. |
| Car Wash | Indicates the presence of a car wash facility at the site. |
| Store SQ FT | Square footage of the store at the site. |
| Lot SQ FT | Total square footage of the lot or site area. |

## 🔧 Technical Architecture

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  Data Acquisition  │────►│   Data Cleaning   │────►│Feature Engineering│────►│  Model Training   │
└───────────────────┘     └───────────────────┘     └───────────────────┘     └───────────────────┘
                                                                                        │
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐               ▼
│  Forecast Output   │◄────│ Forecast Generator│◄────│ Feature Importance │◄────┌───────────────────┐
└───────────────────┘     └───────────────────┘     └───────────────────┘     │ Model Evaluation │
                                                                             └───────────────────┘
```

## ✨ Key Features

- **Multi-modal time series forecasting**: Ensemble approach combining tree-based algorithms (LightGBM, XGBoost, CatBoost), deep learning (LSTM), and classical time series methods (SARIMAX, Prophet)
- **Advanced feature engineering**: Time-based features, lagged variables, rolling statistics, and external data integration
- **Reliable evaluation framework**: Cross-validation with forward chaining for time series data
- **Outlier detection and management**: IQR and Winsorization techniques
- **External data integration**: Weather data from Meteostat with seamless API integration
- **Feature importance analysis**: Quantifies impact of different variables on forecast accuracy
- **Customizable forecast horizons**: Configurable prediction windows (default: 30 days)
- **Production-ready output**: Exportable forecasts in CSV format

## 🛠️ Implementation Details

### 1. Data Acquisition
Loads data from CSV sources with configurable paths.

```python
loader = DataAcquisition(
    fuel_path='/path/to/Fuel_Data.csv',
    site_path='/path/to/SITE_DATA.csv'
)
```

### 2. Data Cleaning
Implements a two-phase cleaning process:
- Basic cleaning: Standardizes column names, removes duplicates, replaces placeholders
- Full cleaning: Handles outliers, filters relevant sites, imputes missing values

```python
# Choose between IQR-based or Winsorization for outlier treatment
fuel_cleaned, site_cleaned = full_cleaner.process(outlier_method="winsor", verbose=True)
```

### 3. Feature Engineering
Creates a rich feature set including:
- Calendar features (day of week, month, holiday indicators)
- Time-lagged variables (1-day and 7-day lags)
- Rolling statistics (7-day moving average, standard deviation)
- Exponential moving averages
- Site characteristics (encoded categorical variables)
- Weather features (temperature, precipitation indicators)

### 4. Model Selection
Supports multiple algorithms with automatic hyperparameter configuration:

| Model Type | Implementation | Best Use Case |
|------------|----------------|---------------|
| LightGBM | `LGBMRegressor` | General purpose, fast training |
| XGBoost | `XGBRegressor` | High accuracy needs |
| CatBoost | `CatBoostRegressor` | Handles categorical features well |
| Ensemble | Averaging of tree models | Improved generalization |
| SARIMAX | `statsmodels.tsa.statespace.sarimax` | Strong seasonality patterns |
| Prophet | Facebook Prophet | Strong trend with multiple seasonality |
| LSTM | TensorFlow/Keras | Complex non-linear patterns |

### 5. Training with Time Series Cross-Validation
Implements forward chaining cross-validation:
- Automatically segments data by site
- Creates temporally consistent train/validation splits
- Evaluates on multiple forecast horizons
- Reports RMSE, SMAPE and accuracy metrics

### 6. Feature Importance Analysis
Quantifies the impact of each feature on prediction accuracy using model-based importance scores.

### 7. Forecast Generation
Generates detailed site-specific forecasts for customizable time horizons (default: 30 days):

- **Site-level granularity**: Each retail location receives its own dedicated forecast series
- **Daily resolution**: Predicts sales values for each day in the forecast horizon
- **Trend visualization**: Automatically generates time-series plots for each site's forecast
- **Forward propagation**: Dynamically updates time-based features and lagged variables
- **Export capability**: Saves forecasts to CSV with site identifiers for operational use

Example forecast for Site 118 (30-day horizon):

```
┌────────────┬─────────────┬─────────┐
│    Date    │ Site ID     │ Sales($)│
├────────────┼─────────────┼─────────┤
│ 2024-01-01 │    118      │  252.14 │
│ 2024-01-02 │    118      │  202.38 │
│ 2024-01-03 │    118      │  209.71 │
│     ...    │    ...      │   ...   │
│ 2024-01-29 │    118      │  274.92 │
│ 2024-01-30 │    118      │  273.65 │
└────────────┴─────────────┴─────────┘
```

Each site's forecast is visualized with daily values, showing patterns, trends, and potential seasonality:

![30-Day Sales Forecast Visualization](./Project_images/Forecast_site_118.png)

## 📏 Performance Metrics

The system evaluates forecasts using multiple metrics:

1. **RMSE (Root Mean Squared Error)**: Measures absolute prediction error in the same units as the target variable
2. **SMAPE (Symmetric Mean Absolute Percentage Error)**: Scale-independent error measurement (0-100%)
3. **Forecast Accuracy**: Derived as (100 - SMAPE)%, providing an intuitive accuracy percentage

## 🚀 Usage Examples

### Basic Usage

```python
# 1. Load and clean data
loader = DataAcquisition('fuel_data.csv', 'site_data.csv')
fuel_raw, site_raw = loader.load_data()
basic_cleaner = BasicCleaner(fuel_raw, site_raw)
fuel_clean, site_clean = basic_cleaner.clean()

# 2. Process and engineer features
full_cleaner = FullCleaner(fuel_clean, site_clean)
fuel_cleaned, site_cleaned = full_cleaner.process(outlier_method="winsor")
fe = FeatureEngineer(fuel_cleaned, site_cleaned)
final_df = fe.engineer()

# 3. Train models (90-day forecast horizon, 3-fold CV)
model = TimeSeriesModel(final_df, forecast_days=90, cv_folds=3, 
                       model_type='lightgbm', 
                       external_features=['avg_temp_f', 'is_rain'])
y_true, y_pred, trained_models, history_data = model.train()

# 4. Evaluate performance
evaluator = ModelEvaluator(y_true, y_pred)
evaluator.report()

# 5. Generate 30-day forecasts
forecast_gen = ForecastGenerator(trained_models, history_data, 
                                model.selected_features, 
                                forecast_days=30, 
                                model_type=model.model_type)
forecast_df = forecast_gen.forecast()
forecast_df.to_csv("sitewise_30_day_forecast.csv", index=False)
```

### Advanced Configuration

```python
# Using ensemble model with custom feature selection
selected_features = [
    'is_weekend', 'is_holiday', 'avg_unit_price', 
    'dayofweek', 'month', 'lag_1', 'lag_7', 
    'rolling_mean_7', 'rolling_std_7', 'ema_sales'
]

model = TimeSeriesModel(
    final_df, 
    forecast_days=60, 
    cv_folds=5,
    model_type='ensemble',  # Combines CatBoost, LightGBM and XGBoost
    external_features=['avg_temp_f', 'is_rain', 'precip_mm']
)
```

## 📚 Dependencies

The system requires the following key packages:

- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn, catboost, lightgbm, xgboost
- **Time Series**: statsmodels, prophet
- **Deep Learning**: tensorflow
- **External Data**: meteostat, holidays
- **Visualization**: matplotlib, seaborn

## 💻 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/fuel-sales-forecasting.git
cd fuel-sales-forecasting

# Install dependencies
pip install -r requirements.txt

# Install additional packages if using all model types
pip install holidays catboost lightgbm xgboost meteostat statsmodels prophet tensorflow
```

## ⚙️ Performance Considerations

- **Memory Usage**: The system processes data in-memory, with typical requirements of 4-8GB RAM for datasets with up to 5 years of daily data across hundreds of sites.
- **Training Time**: Model training scales with the number of sites and CV folds:
  - Tree models: ~5-15 minutes
  - Prophet: ~30-60 minutes
  - LSTM: ~60-120 minutes
- **Forecast Generation**: 30-day forecasts typically require 1-2 minutes per site

## 🔍 Advanced Customization

### Custom Feature Engineering

The FeatureEngineer class can be extended to incorporate additional features:

```python
class EnhancedFeatureEngineer(FeatureEngineer):
    def __init__(self, fuel_df, site_df):
        super().__init__(fuel_df, site_df)
        
    def engineer(self):
        self.pivot_fuel()
        self.merge_site_data()
        self.label_encode()
        self.add_weather_data()
        self.add_custom_features()  # Custom method
        return self.final_df
        
    def add_custom_features(self):
        # Add economic indicators
        self.final_df['gdp_growth'] = 2.1  # Example placeholder
        
        # Add competitor density features
        self.final_df['competitor_count'] = 3  # Example placeholder
        
        # Add promotional period indicators
        self.final_df['is_promo_period'] = self.final_df['date'].dt.month.isin([6, 11, 12]).astype(int)
```

### Hyperparameter Tuning

For production deployment, hyperparameter optimization is recommended:

```python
from sklearn.model_selection import GridSearchCV

# Example for LightGBM
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'num_leaves': [31, 63, 127],
    'min_child_samples': [5, 20, 50]
}

# Create TimeSeriesModel with CV that respects time
model = TimeSeriesModel(final_df, forecast_days=30, cv_folds=3, model_type='lightgbm')

# Grid search with time series CV
grid_search = GridSearchCV(
    model.get_base_model(),  # Method to expose base model
    param_grid,
    cv=model.get_time_cv(),  # Method to expose time-based CV
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```

## 🚀 Production Deployment Recommendations

1. **Model Versioning**: Use MLflow or similar tool to track model versions, parameters, and performance
2. **Automated Retraining**: Implement scheduled retraining pipeline (monthly recommended)
3. **Model Monitoring**: Implement drift detection to identify when model accuracy degrades
4. **Scalable Inference**: For large site counts, parallelize forecast generation
5. **API Interface**: Wrap forecasting functionality in REST API for integration with other systems

## ❓ Troubleshooting

### Common Issues

1. **Missing/NaN Values in Weather Data**
   - Cause: Meteostat API limitations or regional data gaps
   - Solution: Enable more aggressive imputation or use alternative weather API

2. **Zero/Negative Sales Predictions**
   - Cause: Outlier treatment or model limitations
   - Solution: Apply post-processing constraints to enforce non-negative values

3. **High Error Rates for Specific Sites**
   - Cause: Irregular patterns or insufficient historical data
   - Solution: Create site clusters and train separate models for similar sites

## 🤝 Contributing

Contributions to enhance functionality or improve performance are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-enhancement`)
3. Commit changes (`git commit -m 'Add amazing enhancement'`)
4. Push to branch (`git push origin feature/amazing-enhancement`)
5. Open a Pull Request

## 📧 Contact

For any queries regarding the project, please reach out to:
- Email: [Email](praveen.lenkalapelly9@gmail.com)
