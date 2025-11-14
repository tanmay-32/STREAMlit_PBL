
"""
Gaming Cafe Analytics Dashboard
Complete ML Pipeline: Classification | Clustering | Association Rules | Regression | Dynamic Pricing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (silhouette_score, davies_bouldin_score, 
                             mean_squared_error, r2_score, mean_absolute_error)
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Gaming Cafe Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    h1 {
        color: #ffffff;
        font-family: 'Helvetica Neue', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h2, h3 {
        color: #ffffff;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and Header
st.title("🎮 Gaming Cafe Analytics Dashboard")
st.markdown("### Complete ML Pipeline: Classification | Clustering | Association Rules | Regression | Dynamic Pricing")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/controller.png", width=100)
    st.title("⚙️ Dashboard Controls")
    st.markdown("---")

    # Data Upload Section
    st.subheader("📁 Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Use Sample Data", "Upload Custom Data"],
        help="Use sample data or upload your own CSV file"
    )

    uploaded_file = None
    if data_source == "Upload Custom Data":
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="Upload your gaming cafe survey data"
        )

    st.markdown("---")
    st.info("💡 **Tip:** Start with sample data to explore features, then upload your own!")

# Helper Functions
@st.cache_data
def load_data(uploaded_file=None):
    """Load data from file or use sample"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Try to load sample data
        try:
            df = pd.read_csv('gaming_cafe_market_survey_600_responses.csv')
        except:
            st.error("Sample data file not found. Please upload your data.")
            return None
    return df

def download_csv(df, filename):
    """Generate download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def preprocess_data(df):
    """Preprocess data for ML models"""
    df_work = df.copy()

    # Ordinal mappings
    ordinal_mappings = {
        'Q1_Age': {'Under 18': 0, '18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55 and above': 5},
        'Q6_Monthly_Income_AED': {
            'Below 5,000': 0, '5,000 - 10,000': 1, '10,001 - 20,000': 2,
            '20,001 - 35,000': 3, '35,001 - 50,000': 4, 'Above 50,000': 5,
            'Prefer not to say': 2
        },
        'Q11_Play_Video_Games': {
            'No, and not interested': 0, "No, but I'm interested in starting": 1,
            'Yes, rarely (few times a year)': 2, 'Yes, occasionally (few times a month)': 3,
            'Yes, regularly (at least once a week)': 4
        },
        'Q15_Hours_Per_Week': {
            'Less than 2 hours': 0, '2-5 hours': 1, '6-10 hours': 2,
            '11-15 hours': 3, '16-20 hours': 4, 'More than 20 hours': 5
        }
    }

    # Apply ordinal encoding
    for col, mapping in ordinal_mappings.items():
        if col in df_work.columns:
            df_work[col] = df_work[col].map(mapping)

    # Label encode remaining categorical
    le = LabelEncoder()
    for col in df_work.select_dtypes(include=['object']).columns:
        df_work[col] = le.fit_transform(df_work[col].astype(str))

    # Fill missing values
    df_work = df_work.fillna(df_work.median())

    return df_work

# Load Data
df = load_data(uploaded_file)

if df is not None:
    st.success(f"✅ Data loaded successfully! {len(df)} responses")

    # Show data preview
    with st.expander("📊 View Data Sample"):
        st.dataframe(df.head(10), use_container_width=True)
        st.download_button(
            label="📥 Download Full Dataset",
            data=df.to_csv(index=False),
            file_name="gaming_cafe_data.csv",
            mime="text/csv"
        )

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🎯 Clustering & Personas", 
        "🔗 Association Rules",
        "💰 Regression & Pricing",
        "🎛️ Dynamic Pricing Engine"
    ])

    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    with tab1:
        st.header("📊 Data Overview & Key Insights")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Responses",
                value=len(df),
                delta="Sample Size"
            )

        with col2:
            if 'Q45_Interest_In_Concept' in df.columns:
                interested = len(df[~df['Q45_Interest_In_Concept'].str.contains('Not interested', na=False)])
                interest_rate = (interested / len(df)) * 100
                st.metric(
                    label="Interest Rate",
                    value=f"{interest_rate:.1f}%",
                    delta="Positive Signal"
                )

        with col3:
            if 'Q1_Age' in df.columns:
                mode_age = df['Q1_Age'].mode()[0]
                st.metric(
                    label="Primary Age Group",
                    value=mode_age,
                    delta="Target Demographic"
                )

        with col4:
            if 'Q6_Monthly_Income_AED' in df.columns:
                mode_income = df['Q6_Monthly_Income_AED'].mode()[0]
                st.metric(
                    label="Common Income",
                    value=mode_income,
                    delta="Spending Power"
                )

        st.markdown("---")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Age Distribution")
            if 'Q1_Age' in df.columns:
                age_dist = df['Q1_Age'].value_counts().sort_index()
                fig = px.bar(
                    x=age_dist.index, 
                    y=age_dist.values,
                    labels={'x': 'Age Group', 'y': 'Count'},
                    color=age_dist.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Interest Level Distribution")
            if 'Q45_Interest_In_Concept' in df.columns:
                interest_dist = df['Q45_Interest_In_Concept'].value_counts()
                fig = px.pie(
                    values=interest_dist.values,
                    names=interest_dist.index,
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 2: CLUSTERING & PERSONAS
    # ========================================================================
    with tab2:
        st.header("🎯 Customer Clustering & Persona Analysis")

        # Sidebar controls for clustering
        with st.sidebar:
            st.markdown("### 🎯 Clustering Settings")
            n_clusters = st.slider("Number of Clusters (K)", 2, 10, 5, help="Choose optimal number of customer segments")
            clustering_method = st.selectbox("Clustering Method", ["K-Means", "Gaussian Mixture Model"])

        # Feature selection for clustering
        clustering_features = [
            'Q1_Age', 'Q6_Monthly_Income_AED', 'Q11_Play_Video_Games',
            'Q15_Hours_Per_Week', 'Q37_Total_WTP_Per_Visit_AED',
            'Q38_Price_Sensitivity', 'Q26_Food_Quality_Importance',
            'Q45_Interest_In_Concept', 'Q47_Expected_Visit_Frequency',
            'Q21_Social_Aspect_Importance'
        ]

        # Check if all features exist
        clustering_features = [f for f in clustering_features if f in df.columns]

        if len(clustering_features) > 5:
            df_processed = preprocess_data(df[clustering_features])

            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_processed)

            # Apply clustering
            if clustering_method == "K-Means":
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            else:
                model = GaussianMixture(n_components=n_clusters, random_state=42)

            clusters = model.fit_predict(X_scaled)
            df['Cluster'] = clusters

            # Calculate metrics
            silhouette = silhouette_score(X_scaled, clusters)
            davies_bouldin = davies_bouldin_score(X_scaled, clusters)

            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score", f"{silhouette:.3f}", help="Higher is better (max 1.0)")
            with col2:
                st.metric("Davies-Bouldin Score", f"{davies_bouldin:.3f}", help="Lower is better")
            with col3:
                st.metric("Number of Clusters", n_clusters)

            st.markdown("---")

            # PCA for visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            df['PCA1'] = X_pca[:, 0]
            df['PCA2'] = X_pca[:, 1]

            # Cluster visualization
            st.subheader("Customer Segments Visualization")
            fig = px.scatter(
                df, x='PCA1', y='PCA2', color='Cluster',
                title=f"{clustering_method} Clustering (PCA Projection)",
                color_continuous_scale='viridis',
                hover_data=['Q1_Age', 'Q6_Monthly_Income_AED'] if all(c in df.columns for c in ['Q1_Age', 'Q6_Monthly_Income_AED']) else None
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Cluster distribution
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Cluster Size Distribution")
                cluster_counts = df['Cluster'].value_counts().sort_index()
                fig = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={'x': 'Cluster', 'y': 'Customer Count'},
                    color=cluster_counts.values,
                    color_continuous_scale='blues'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Cluster Characteristics")
                # Calculate mean characteristics
                cluster_profile = df.groupby('Cluster')[clustering_features[:5]].mean()
                st.dataframe(cluster_profile.style.background_gradient(cmap='RdYlGn'), use_container_width=True)

            # Download results
            st.download_button(
                label="📥 Download Clustering Results",
                data=df.to_csv(index=False),
                file_name="clustering_results.csv",
                mime="text/csv"
            )
        else:
            st.warning("Not enough features available for clustering analysis.")

    # ========================================================================
    # TAB 3: ASSOCIATION RULES
    # ========================================================================
    with tab3:
        st.header("🔗 Association Rule Mining")

        # Sidebar controls
        with st.sidebar:
            st.markdown("### 🔗 Association Rules Settings")
            min_support = st.slider("Minimum Support (%)", 1, 50, 10, help="Minimum % of transactions") / 100
            min_confidence = st.slider("Minimum Confidence (%)", 10, 100, 70, help="Rule reliability %") / 100
            top_n_rules = st.slider("Top N Rules to Display", 5, 50, 10)

        # Check for required columns
        if 'Q13_Game_Types_Preferred' in df.columns and 'Q23_Leisure_Venues_Visited' in df.columns:

            # Extract transactions
            transactions = []
            for idx, row in df.iterrows():
                items = []
                if pd.notna(row['Q13_Game_Types_Preferred']):
                    items.extend([x.strip() for x in str(row['Q13_Game_Types_Preferred']).split(';')])
                if pd.notna(row['Q23_Leisure_Venues_Visited']):
                    items.extend([x.strip() for x in str(row['Q23_Leisure_Venues_Visited']).split(';')])
                if items:
                    transactions.append(items)

            if len(transactions) > 0:
                # Transform to binary
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

                # Apply Apriori
                frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

                if len(frequent_itemsets) > 0:
                    # Generate rules
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

                    if len(rules) > 0:
                        rules = rules.sort_values('confidence', ascending=False).head(top_n_rules)

                        # Display metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Frequent Itemsets", len(frequent_itemsets))
                        with col2:
                            st.metric("Association Rules", len(rules))
                        with col3:
                            st.metric("Avg Confidence", f"{rules['confidence'].mean():.2%}")

                        st.markdown("---")

                        # Display rules
                        st.subheader(f"Top {len(rules)} Association Rules")

                        # Format rules for display
                        rules_display = rules.copy()
                        rules_display['antecedents'] = rules_display['antecedents'].apply(lambda x: ', '.join(list(x)))
                        rules_display['consequents'] = rules_display['consequents'].apply(lambda x: ', '.join(list(x)))
                        rules_display['support'] = rules_display['support'].apply(lambda x: f"{x:.1%}")
                        rules_display['confidence'] = rules_display['confidence'].apply(lambda x: f"{x:.1%}")
                        rules_display['lift'] = rules_display['lift'].apply(lambda x: f"{x:.2f}")

                        st.dataframe(
                            rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
                            use_container_width=True
                        )

                        # Visualize rules
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Support vs Confidence")
                            fig = px.scatter(
                                rules, x='support', y='confidence', size='lift',
                                hover_data=['antecedents', 'consequents'],
                                color='lift', color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.subheader("Lift Distribution")
                            fig = px.histogram(rules, x='lift', nbins=20, color_discrete_sequence=['#667eea'])
                            st.plotly_chart(fig, use_container_width=True)

                        # Download rules
                        st.download_button(
                            label="📥 Download Association Rules",
                            data=rules_display.to_csv(index=False),
                            file_name="association_rules.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"No rules found with confidence ≥ {min_confidence:.0%}. Try lowering the threshold.")
                else:
                    st.warning(f"No frequent itemsets found with support ≥ {min_support:.0%}. Try lowering the threshold.")
            else:
                st.error("No valid transactions found in the data.")
        else:
            st.error("Required columns not found: Q13_Game_Types_Preferred and Q23_Leisure_Venues_Visited")

    # ========================================================================
    # TAB 4: REGRESSION & PRICING
    # ========================================================================
    with tab4:
        st.header("💰 Regression Analysis & Price Prediction")

        # Sidebar controls
        with st.sidebar:
            st.markdown("### 💰 Regression Settings")
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            selected_models = st.multiselect(
                "Select Models to Compare",
                ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest", "Gradient Boosting"],
                default=["Linear Regression", "Random Forest", "Gradient Boosting"]
            )

        # Check for target variable
        target_col = 'Q37_Total_WTP_Per_Visit_AED'

        if target_col in df.columns:
            # Prepare data
            predictor_features = [
                'Q1_Age', 'Q6_Monthly_Income_AED', 'Q11_Play_Video_Games',
                'Q15_Hours_Per_Week', 'Q38_Price_Sensitivity',
                'Q26_Food_Quality_Importance', 'Q45_Interest_In_Concept',
                'Q47_Expected_Visit_Frequency', 'Q21_Social_Aspect_Importance'
            ]

            predictor_features = [f for f in predictor_features if f in df.columns]

            if len(predictor_features) > 3:
                # Convert target to numeric
                spending_mapping = {
                    '50-100 AED': 75, '101-150 AED': 125, '151-200 AED': 175,
                    '201-300 AED': 250, '301-400 AED': 350, 'Above 400 AED': 450
                }

                df_reg = df.copy()
                df_reg[target_col] = df_reg[target_col].map(spending_mapping)

                # Preprocess
                df_processed = preprocess_data(df_reg[predictor_features + [target_col]])

                X = df_processed[predictor_features]
                y = df_processed[target_col]

                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Model dictionary
                models_dict = {
                    "Linear Regression": LinearRegression(),
                    "Ridge": Ridge(alpha=1.0),
                    "Lasso": Lasso(alpha=1.0),
                    "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
                }

                # Train and evaluate models
                results = {}

                for name in selected_models:
                    if name in models_dict:
                        model = models_dict[name]

                        # Train
                        if name in ["Linear Regression", "Ridge", "Lasso"]:
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                        # Calculate metrics
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        mae = mean_absolute_error(y_test, y_pred)

                        results[name] = {
                            'R² Score': r2,
                            'RMSE': rmse,
                            'MAE': mae,
                            'predictions': y_pred
                        }

                # Display results
                st.subheader("Model Performance Comparison")

                # Create comparison dataframe
                comparison_df = pd.DataFrame({
                    'Model': list(results.keys()),
                    'R² Score': [results[m]['R² Score'] for m in results.keys()],
                    'RMSE (AED)': [results[m]['RMSE'] for m in results.keys()],
                    'MAE (AED)': [results[m]['MAE'] for m in results.keys()]
                })

                st.dataframe(
                    comparison_df.style.background_gradient(subset=['R² Score'], cmap='RdYlGn')
                                     .background_gradient(subset=['RMSE (AED)'], cmap='RdYlGn_r')
                                     .format({'R² Score': '{:.3f}', 'RMSE (AED)': '{:.2f}', 'MAE (AED)': '{:.2f}'}),
                    use_container_width=True
                )

                # Visualizations
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("R² Score Comparison")
                    fig = px.bar(
                        comparison_df, x='Model', y='R² Score',
                        color='R² Score', color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("RMSE Comparison")
                    fig = px.bar(
                        comparison_df, x='Model', y='RMSE (AED)',
                        color='RMSE (AED)', color_continuous_scale='reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Best model analysis
                best_model_name = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']
                st.success(f"🏆 Best Model: **{best_model_name}** (R² = {results[best_model_name]['R² Score']:.3f})")

                # Predicted vs Actual
                st.subheader(f"{best_model_name}: Predicted vs Actual")

                pred_actual_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': results[best_model_name]['predictions']
                })

                fig = px.scatter(
                    pred_actual_df, x='Actual', y='Predicted',
                    trendline="ols", trendline_color_override="red"
                )
                fig.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines', name='Perfect Prediction',
                    line=dict(dash='dash', color='green')
                ))
                st.plotly_chart(fig, use_container_width=True)

                # Download results
                st.download_button(
                    label="📥 Download Model Comparison",
                    data=comparison_df.to_csv(index=False),
                    file_name="regression_comparison.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Not enough predictor features available.")
        else:
            st.error(f"Target variable '{target_col}' not found in dataset.")

    # ========================================================================
    # TAB 5: DYNAMIC PRICING ENGINE
    # ========================================================================
    with tab5:
        st.header("🎛️ Dynamic Pricing Engine")
        st.markdown("### Personalized Pricing Based on Customer Attributes")

        # Sidebar controls
        with st.sidebar:
            st.markdown("### 🎛️ Pricing Parameters")
            base_price = st.number_input("Base Price (AED)", 50, 500, 150, step=10)
            max_discount = st.slider("Max Loyalty Discount (%)", 0, 50, 20) / 100
            price_sensitivity_weight = st.slider("Price Sensitivity Weight", 0.0, 1.0, 0.5, 0.1)

        # Calculate loyalty scores and dynamic pricing
        if all(col in df.columns for col in ['Q17_Gaming_Cafe_Visits_Past_12mo', 'Q47_Expected_Visit_Frequency', 'Q45_Interest_In_Concept']):

            # Encode if needed
            df_price = preprocess_data(df[[
                'Q17_Gaming_Cafe_Visits_Past_12mo',
                'Q47_Expected_Visit_Frequency',
                'Q45_Interest_In_Concept',
                'Q11_Play_Video_Games',
                'Q38_Price_Sensitivity'
            ]].copy() if all(c in df.columns for c in ['Q11_Play_Video_Games', 'Q38_Price_Sensitivity']) else df[[
                'Q17_Gaming_Cafe_Visits_Past_12mo',
                'Q47_Expected_Visit_Frequency',
                'Q45_Interest_In_Concept'
            ]].copy())

            # Calculate loyalty score (0-100)
            df_price['Loyalty_Score'] = (
                df_price['Q17_Gaming_Cafe_Visits_Past_12mo'] * 30 +
                df_price['Q47_Expected_Visit_Frequency'] * 25 +
                df_price['Q45_Interest_In_Concept'] * 20
            ).clip(0, 100)

            # Loyalty tiers
            df_price['Loyalty_Tier'] = pd.cut(
                df_price['Loyalty_Score'],
                bins=[0, 40, 60, 80, 100],
                labels=['Bronze', 'Silver', 'Gold', 'Platinum']
            )

            # Calculate dynamic price
            df_price['Loyalty_Discount'] = (df_price['Loyalty_Score'] / 100) * max_discount
            df_price['Dynamic_Price'] = base_price * (1 - df_price['Loyalty_Discount'])

            if 'Q38_Price_Sensitivity' in df_price.columns:
                # Adjust for price sensitivity
                price_adj = 1 + (df_price['Q38_Price_Sensitivity'] - 2.5) * price_sensitivity_weight * 0.1
                df_price['Dynamic_Price'] = df_price['Dynamic_Price'] * price_adj

            df_price['Savings'] = base_price - df_price['Dynamic_Price']
            df_price['Discount_Pct'] = (df_price['Savings'] / base_price) * 100

            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Base Price", f"{base_price} AED")
            with col2:
                avg_price = df_price['Dynamic_Price'].mean()
                st.metric("Avg Dynamic Price", f"{avg_price:.2f} AED")
            with col3:
                avg_discount = df_price['Discount_Pct'].mean()
                st.metric("Avg Discount", f"{avg_discount:.1f}%")
            with col4:
                total_revenue = df_price['Dynamic_Price'].sum()
                st.metric("Total Revenue Potential", f"{total_revenue:,.0f} AED")

            st.markdown("---")

            # Visualizations
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Price Distribution by Loyalty Tier")
                fig = px.box(
                    df_price, x='Loyalty_Tier', y='Dynamic_Price',
                    color='Loyalty_Tier',
                    color_discrete_map={'Bronze': '#CD7F32', 'Silver': '#C0C0C0', 
                                       'Gold': '#FFD700', 'Platinum': '#E5E4E2'}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Loyalty Score Distribution")
                fig = px.histogram(
                    df_price, x='Loyalty_Score', nbins=30,
                    color_discrete_sequence=['#667eea']
                )
                st.plotly_chart(fig, use_container_width=True)

            # Customer tier distribution
            st.subheader("Customer Distribution by Loyalty Tier")
            tier_dist = df_price['Loyalty_Tier'].value_counts()

            col1, col2 = st.columns([1, 2])

            with col1:
                fig = px.pie(
                    values=tier_dist.values,
                    names=tier_dist.index,
                    color=tier_dist.index,
                    color_discrete_map={'Bronze': '#CD7F32', 'Silver': '#C0C0C0',
                                       'Gold': '#FFD700', 'Platinum': '#E5E4E2'},
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Tier summary table
                tier_summary = df_price.groupby('Loyalty_Tier').agg({
                    'Dynamic_Price': ['mean', 'min', 'max'],
                    'Discount_Pct': 'mean',
                    'Loyalty_Score': 'mean'
                }).round(2)

                tier_summary.columns = ['Avg Price', 'Min Price', 'Max Price', 'Avg Discount %', 'Avg Loyalty Score']
                st.dataframe(tier_summary, use_container_width=True)

            # Sample rate card
            st.subheader("📋 Sample Digital Rate Card (First 10 Customers)")
            rate_card = df_price[['Loyalty_Score', 'Loyalty_Tier', 'Dynamic_Price', 'Discount_Pct', 'Savings']].head(10)
            rate_card.index = [f"Customer {i+1}" for i in range(len(rate_card))]

            st.dataframe(
                rate_card.style.background_gradient(subset=['Loyalty_Score'], cmap='RdYlGn')
                              .format({'Dynamic_Price': '{:.2f} AED', 'Discount_Pct': '{:.1f}%', 
                                      'Savings': '{:.2f} AED', 'Loyalty_Score': '{:.0f}'}),
                use_container_width=True
            )

            # Download pricing results
            full_rate_card = df_price[['Loyalty_Score', 'Loyalty_Tier', 'Dynamic_Price', 'Discount_Pct', 'Savings']]
            full_rate_card.index = [f"Customer {i+1}" for i in range(len(full_rate_card))]

            st.download_button(
                label="📥 Download Complete Digital Rate Card",
                data=full_rate_card.to_csv(),
                file_name="digital_rate_card.csv",
                mime="text/csv"
            )
        else:
            st.warning("Required columns for pricing analysis not found in dataset.")

else:
    st.warning("⚠️ Please upload data or ensure sample data file is available.")
    st.info("💡 The app expects a file named 'gaming_cafe_market_survey_600_responses.csv' in the same directory.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white;'>
    <p>🎮 Gaming Cafe Analytics Dashboard | Built with Streamlit & ML</p>
    <p>Complete Pipeline: Clustering | Association Rules | Regression | Dynamic Pricing</p>
</div>
""", unsafe_allow_html=True)
