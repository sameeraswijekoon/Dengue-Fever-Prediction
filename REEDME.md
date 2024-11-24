# 🦟 Dengue Fever Prediction System: Sri Lanka Case Study

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sameeraswijekoon/Dengue-Fever-Prediction.git)

## 📊 Executive Summary

This project develops a machine learning-based system to predict dengue fever outbreaks in Sri Lanka using environmental data, particularly rainfall patterns. The system aims to provide early warnings to public health officials, enabling proactive measures against potential outbreaks.

### 🎯 Key Objectives
- Predict dengue outbreak risks using rainfall data
- Provide district-wise analysis across Sri Lanka
- Enable data-driven public health decision making
- Create an accessible prediction tool for health officials

## 🔬 Technical Implementation

### Data Processing Pipeline

```python
# Core data processing workflow
1. Data Collection
   └── Sources: Kaggle + Sri Lanka Meteorological Department
2. Data Preprocessing
   ├── Clean missing values
   ├── Normalize rainfall data
   └── Encode district information
3. Feature Engineering
   ├── Monthly rainfall metrics
   ├── District encoding
   └── Temporal features
4. Model Development
   ├── Linear Regression baseline
   └── Decision Tree alternative
```

### 🛠️ Technical Architecture

#### Core Components
1. **Data Preprocessing Module**
   ```python
   import pandas as pd
   
   def preprocess_data(df):
       # Handle missing values
       df.dropna(inplace=True)
       
       # Encode districts
       district_mapping = {
           'Colombo': 1, 'Gampaha': 2, 'Kalutara': 3,
           # ... other districts
       }
       df['District_Code'] = df['District'].map(district_mapping)
       return df
   ```

2. **Model Training Pipeline**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LinearRegression
   
   def train_monthly_models(df, months, rain_columns):
       models = {}
       for month, rain_col in zip(months, rain_columns):
           X = df[['District_Code', 'Year', rain_col]]
           y = df[month]
           
           # Split and train
           X_train, X_test, y_train, y_test = train_test_split(
               X, y, test_size=0.2, random_state=42
           )
           
           model = LinearRegression()
           model.fit(X_train, y_train)
           models[month] = model
       return models
   ```

## 📈 Performance Analysis

### Model Evaluation

Monthly MSE Results:
```
April:     2,901 (Best Performance)
March:     6,479
May:       7,090
February:  9,303
June:      25,866
September: 25,172
January:   41,039
October:   41,782
July:      49,032
August:    49,383
December:  192,109
November:  359,559 (Needs Improvement)
```

### Key Findings
1. **Seasonal Variation**
   - Best predictions: March-May period
   - Challenging predictions: November-December period
   - Suggests stronger rainfall-dengue correlation in certain seasons

2. **District-wise Performance**
   - Urban areas show more consistent predictions
   - Rural areas have higher variance in prediction accuracy

## 🔍 Critical Analysis

### Strengths
1. **Simplicity**: Linear regression provides interpretable results
2. **Scalability**: Easy to deploy and maintain
3. **Real-time capability**: Can process new data quickly

### Limitations and Future Work

1. **Data Constraints**
   ```python
   # Current features
   features = ['rainfall', 'district', 'year']
   
   # Proposed extensions
   additional_features = [
       'temperature',
       'humidity',
       'population_density',
       'previous_outbreaks',
       'mosquito_breeding_sites'
   ]
   ```

2. **Model Improvements**
   - Implement ensemble methods
   - Add time series analysis
   - Include spatial correlation

3. **Technical Debt**
   - Need automated data pipeline
   - Require robust error handling
   - Missing validation framework

## 🚀 Usage Guide

### Setup Instructions
```bash
# Clone repository
git clone https://github.com/sameeraswijekoon/Dengue-Fever-Prediction.git
cd Dengue-Fever-Prediction

# Install dependencies
pip install -r requirements.txt

# Run prediction system
python dengue_fever_prediction.py
```

### Making Predictions
```python
# Example usage
district = "Colombo"
year = 2024
rainfall = 400
month = "Jan"

prediction = predict_dengue_cases(
    district=district,
    year=year,
    rainfall=rainfall,
    month=month
)
```

## 📝 Recommendations

1. **Data Enhancement**
   - Integrate real-time weather data
   - Add historical outbreak patterns
   - Include socioeconomic factors

2. **Model Improvements**
   - Implement ensemble methods
   - Add cross-validation
   - Develop confidence intervals

3. **System Extensions**
   - Create web-based interface
   - Add automated reporting
   - Implement alert system

## 🔮 Future Roadmap

1. **Q2 2024**
   - Integrate temperature data
   - Improve district-level accuracy

2. **Q3 2024**
   - Deploy web interface
   - Add real-time predictions

3. **Q4 2024**
   - Implement ensemble models
   - Release mobile application

## 📚 References

1. Kaggle Dataset: "Sri Lanka Dengue Data 2019-2021 Insights"
2. Sri Lanka Meteorological Department (Meteo.gov.lk)
3. World Health Organization Dengue Guidelines

---

<div align="center">
Developed with ❤️ for public health in Sri Lanka
</div>
