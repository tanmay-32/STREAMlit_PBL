# 🎮 Gaming Cafe Analytics Dashboard

A comprehensive Streamlit dashboard for gaming cafe market analysis with complete ML pipeline.

## Features

- 📊 **Data Overview**: Interactive visualizations and key metrics
- 🎯 **Clustering & Personas**: K-Means and GMM clustering with adjustable parameters
- 🔗 **Association Rules**: Apriori algorithm with customizable support/confidence
- 💰 **Regression Analysis**: Multiple models comparison for price prediction
- 🎛️ **Dynamic Pricing**: AI-powered personalized pricing engine

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the dashboard
streamlit run app.py
```

## Data Requirements

Place your CSV file named `gaming_cafe_market_survey_600_responses.csv` in the same directory, or upload custom data through the app interface.

### Required Columns:
- Q1_Age
- Q6_Monthly_Income_AED
- Q11_Play_Video_Games
- Q13_Game_Types_Preferred
- Q15_Hours_Per_Week
- Q17_Gaming_Cafe_Visits_Past_12mo
- Q21_Social_Aspect_Importance
- Q23_Leisure_Venues_Visited
- Q26_Food_Quality_Importance
- Q37_Total_WTP_Per_Visit_AED
- Q38_Price_Sensitivity
- Q45_Interest_In_Concept
- Q47_Expected_Visit_Frequency

## Features by Tab

### 📊 Overview
- Total responses and key metrics
- Age and interest level distributions
- Quick insights dashboard

### 🎯 Clustering & Personas
- Adjustable number of clusters (2-10)
- K-Means or Gaussian Mixture Model
- PCA visualization
- Downloadable results

### 🔗 Association Rules
- Adjustable support and confidence thresholds
- Top N rules display
- Interactive visualizations
- CSV export

### 💰 Regression & Pricing
- Multiple model comparison (Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting)
- Adjustable test size
- R², RMSE, MAE metrics
- Predicted vs Actual plots
- Downloadable comparison results

### 🎛️ Dynamic Pricing Engine
- Loyalty score calculation
- 4-tier system (Bronze/Silver/Gold/Platinum)
- Adjustable pricing parameters
- Digital rate card generation
- Revenue optimization insights

## Customization

### Sidebar Controls:
- Upload custom data or use sample
- Adjust clustering parameters
- Set association rule thresholds
- Configure regression test size
- Customize pricing parameters

### Export Options:
- Download datasets
- Export clustering results
- Save association rules
- Get model comparisons
- Generate rate cards

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Machine Learning**: Scikit-learn, MLxtend
- **Clustering**: K-Means, Gaussian Mixture Model
- **Association Rules**: Apriori algorithm
- **Regression**: Multiple algorithms with comparison

## Dashboard Sections

1. **Overview**: Quick insights and data exploration
2. **Clustering**: Customer segmentation analysis
3. **Association Rules**: Market basket analysis
4. **Regression**: Price prediction modeling
5. **Dynamic Pricing**: Personalized pricing engine

## Performance Metrics

- **Clustering**: Silhouette Score, Davies-Bouldin Index
- **Regression**: R², RMSE, MAE
- **Association Rules**: Support, Confidence, Lift
- **Pricing**: Revenue potential, discount optimization

## Author

Gaming Cafe Analytics System
Built with Streamlit, Plotly, and Scikit-learn

## License

MIT License - Feel free to modify and use for your projects!
