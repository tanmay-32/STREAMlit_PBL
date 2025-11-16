"""
Gaming Cafe Analytics Dashboard - FINAL FIXED VERSION
Clustering Error Completely Resolved
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score,
    mean_squared_error, r2_score, mean_absolute_error
)

# Association Rules
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

# Custom CSS
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #f0f2f6; border-radius: 10px 10px 0px 0px;
        padding: 10px 20px; font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
    }
    h1, h2, h3 {color: #ffffff;}
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 10px; border: none; padding: 10px 20px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎮 Gaming Cafe Analytics Dashboard")
st.markdown("### Complete ML Pipeline: Classification | Clustering | Association Rules | Regression | Dynamic Pricing")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/controller.png", width=100)
    st.title("⚙️ Dashboard Controls")
    st.markdown("---")

    st.subheader("📁 Data Source")
    data_source = st.radio(
        "Choose data source:",
        ["Use Sample Data", "Upload Custom Data"],
        help="Use sample data or upload your own CSV file"
    )

    uploaded_file = None
    if data_source == "Upload Custom Data":
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    st.markdown("---")
    st.info("💡 **Tip:** Upload your data or use sample dataset")

# Helper Functions
@st.cache_data
def load_data(uploaded_file=None):
    """Load data from file or use sample"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            url = "https://raw.githubusercontent.com/tanmay-32/streamlit_pbl/main/gaming_cafe_market_survey_600_responses.csv"
            df = pd.read_csv(url)
        except:
            try:
                df = pd.read_csv('gaming_cafe_market_survey_600_responses.csv')
            except:
                st.error("Sample data not found. Please upload your CSV file.")
                return None
    return df

def preprocess_data(df):
    """Preprocess data for ML models"""
    df_work = df.copy()

    ordinal_mappings = {
        'Q1_Age': {'Under 18': 0, '18-24': 1, '25-34': 2, '35-44': 3, '45-54': 4, '55 and above': 5},
        'Q6_Monthly_Income_AED': {
            'Below 5,000': 0, '5,000 - 10,000': 1, '10,001 - 20,000': 2,
            '20,001 - 35,000': 3, '35,001 - 50,000': 4, 'Above 50,000': 5, 'Prefer not to say': 2
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

    for col, mapping in ordinal_mappings.items():
        if col in df_work.columns:
            df_work[col] = df_work[col].map(mapping)

    le = LabelEncoder()
    for col in df_work.select_dtypes(include=['object']).columns:
        try:
            df_work[col] = le.fit_transform(df_work[col].astype(str))
        except:
            pass

    df_work = df_work.fillna(df_work.median(numeric_only=True))

    return df_work

# Load Data
df = load_data(uploaded_file)

if df is not None:
    st.success(f"✅ Data loaded successfully! {len(df)} responses")

    with st.expander("📊 View Data Sample"):
        st.dataframe(df.head(10), use_container_width=True)
        st.download_button(
            label="📥 Download Full Dataset",
            data=df.to_csv(index=False),
            file_name="gaming_cafe_data.csv",
            mime="text/csv"
        )

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "🎯 Classification",
        "🔍 Clustering", 
        "🔗 Association Rules",
        "💰 Regression",
        "🎛️ Dynamic Pricing"
    ])

    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    with tab1:
        st.header("📊 Data Overview & Key Insights")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Responses", len(df))

        with col2:
            if 'Q45_Interest_In_Concept' in df.columns:
                interested = len(df[~df['Q45_Interest_In_Concept'].str.contains('Not interested', na=False)])
                interest_rate = (interested / len(df)) * 100
                st.metric("Interest Rate", f"{interest_rate:.1f}%")

        with col3:
            if 'Q1_Age' in df.columns:
                mode_age = df['Q1_Age'].mode()[0] if len(df['Q1_Age'].mode()) > 0 else "N/A"
                st.metric("Primary Age Group", mode_age)

        with col4:
            if 'Q6_Monthly_Income_AED' in df.columns:
                mode_income = df['Q6_Monthly_Income_AED'].mode()[0] if len(df['Q6_Monthly_Income_AED'].mode()) > 0 else "N/A"
                st.metric("Common Income", mode_income)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Age Distribution")
            if 'Q1_Age' in df.columns:
                age_dist = df['Q1_Age'].value_counts().sort_index()
                fig = px.bar(x=age_dist.index, y=age_dist.values,
                           labels={'x': 'Age Group', 'y': 'Count'},
                           color=age_dist.values, color_continuous_scale='viridis')
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Interest Level Distribution")
            if 'Q45_Interest_In_Concept' in df.columns:
                interest_dist = df['Q45_Interest_In_Concept'].value_counts()
                fig = px.pie(values=interest_dist.values, names=interest_dist.index,
                           hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    # ========================================================================
    # TAB 2: CLASSIFICATION
    # ========================================================================
    with tab2:
        st.header("🎯 Classification Analysis")
        st.markdown("### Predict Customer Interest Level")

        with st.sidebar:
            st.markdown("### 🎯 Classification Settings")
            test_size_class = st.slider("Test Size (%)", 10, 40, 20, key="test_class") / 100
            selected_classifiers = st.multiselect(
                "Select Models",
                ["Logistic Regression", "Decision Tree", "Random Forest", 
                 "Gradient Boosting", "SVM", "KNN", "Naive Bayes"],
                default=["Logistic Regression", "Random Forest", "Gradient Boosting"]
            )

        target_col_class = 'Q45_Interest_In_Concept'

        if target_col_class in df.columns and len(selected_classifiers) > 0:
            try:
                predictor_features_class = [
                    'Q1_Age', 'Q2_Gender', 'Q6_Monthly_Income_AED',
                    'Q11_Play_Video_Games', 'Q15_Hours_Per_Week',
                    'Q21_Social_Aspect_Importance', 'Q26_Food_Quality_Importance',
                    'Q37_Total_WTP_Per_Visit_AED', 'Q38_Price_Sensitivity'
                ]

                predictor_features_class = [f for f in predictor_features_class if f in df.columns]

                if len(predictor_features_class) > 3:
                    df_class = df.copy()

                    df_class['Interest_Binary'] = df_class[target_col_class].apply(
                        lambda x: 1 if 'Extremely' in str(x) or 'Very' in str(x) else 0
                    )

                    df_processed_class = preprocess_data(df_class[predictor_features_class + ['Interest_Binary']])
                    df_processed_class = df_processed_class.select_dtypes(include=[np.number])

                    X = df_processed_class[predictor_features_class]
                    y = df_processed_class['Interest_Binary']

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size_class, random_state=42
                    )

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    classifiers_dict = {
                        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
                        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                        "SVM": SVC(random_state=42, probability=True),
                        "KNN": KNeighborsClassifier(n_neighbors=5),
                        "Naive Bayes": GaussianNB()
                    }

                    results_class = {}

                    for name in selected_classifiers:
                        if name in classifiers_dict:
                            model = classifiers_dict[name]

                            if name in ["Logistic Regression", "SVM", "KNN", "Naive Bayes"]:
                                model.fit(X_train_scaled, y_train)
                                y_pred = model.predict(X_test_scaled)
                            else:
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)

                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
                            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

                            results_class[name] = {
                                'Accuracy': accuracy,
                                'Precision': precision,
                                'Recall': recall,
                                'F1-Score': f1,
                                'predictions': y_pred
                            }

                    st.subheader("📊 Model Performance Comparison")

                    comparison_df_class = pd.DataFrame({
                        'Model': list(results_class.keys()),
                        'Accuracy': [results_class[m]['Accuracy'] for m in results_class.keys()],
                        'Precision': [results_class[m]['Precision'] for m in results_class.keys()],
                        'Recall': [results_class[m]['Recall'] for m in results_class.keys()],
                        'F1-Score': [results_class[m]['F1-Score'] for m in results_class.keys()]
                    })

                    st.dataframe(
                        comparison_df_class.style.background_gradient(cmap='RdYlGn')
                                                 .format({
                                                     'Accuracy': '{:.4f}',
                                                     'Precision': '{:.4f}',
                                                     'Recall': '{:.4f}',
                                                     'F1-Score': '{:.4f}'
                                                 }),
                        use_container_width=True
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Accuracy Comparison")
                        fig = px.bar(comparison_df_class, x='Model', y='Accuracy',
                                   color='Accuracy', color_continuous_scale='viridis',
                                   text='Accuracy')
                        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("F1-Score Comparison")
                        fig = px.bar(comparison_df_class, x='Model', y='F1-Score',
                                   color='F1-Score', color_continuous_scale='blues',
                                   text='F1-Score')
                        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📈 All Metrics Comparison")

                    fig = go.Figure()

                    for model in results_class.keys():
                        fig.add_trace(go.Scatterpolar(
                            r=[results_class[model]['Accuracy'],
                               results_class[model]['Precision'],
                               results_class[model]['Recall'],
                               results_class[model]['F1-Score']],
                            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                            fill='toself',
                            name=model
                        ))

                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True,
                        height=500
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    best_model_class = comparison_df_class.loc[comparison_df_class['Accuracy'].idxmax(), 'Model']
                    st.success(f"🏆 Best Model: **{best_model_class}** (Accuracy = {results_class[best_model_class]['Accuracy']:.4f})")

                    st.subheader(f"🎯 Confusion Matrix - {best_model_class}")

                    cm = confusion_matrix(y_test, results_class[best_model_class]['predictions'])

                    fig = px.imshow(cm, 
                                   labels=dict(x="Predicted", y="Actual", color="Count"),
                                   x=['Not Interested', 'Interested'],
                                   y=['Not Interested', 'Interested'],
                                   color_continuous_scale='Blues',
                                   text_auto=True)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    st.download_button(
                        label="📥 Download Classification Results",
                        data=comparison_df_class.to_csv(index=False),
                        file_name="classification_results.csv",
                        mime="text/csv"
                    )

                else:
                    st.warning("Not enough predictor features available.")

            except Exception as e:
                st.error(f"Error in classification: {str(e)}")
        else:
            if len(selected_classifiers) == 0:
                st.info("Please select at least one classification model from the sidebar.")
            else:
                st.error(f"Target variable not found.")

    # ========================================================================
    # TAB 3: CLUSTERING - COMPLETELY FIXED
    # ========================================================================
    with tab3:
        st.header("🔍 Customer Clustering & Persona Analysis")

        with st.sidebar:
            st.markdown("### 🔍 Clustering Settings")
            n_clusters = st.slider("Number of Clusters (K)", 2, 10, 5, key="n_clusters")
            clustering_method = st.selectbox("Method", ["K-Means", "Gaussian Mixture Model"])

        clustering_features = [
            'Q1_Age', 'Q6_Monthly_Income_AED', 'Q11_Play_Video_Games',
            'Q15_Hours_Per_Week', 'Q37_Total_WTP_Per_Visit_AED',
            'Q38_Price_Sensitivity', 'Q26_Food_Quality_Importance',
            'Q45_Interest_In_Concept', 'Q47_Expected_Visit_Frequency',
            'Q21_Social_Aspect_Importance'
        ]

        clustering_features = [f for f in clustering_features if f in df.columns]

        if len(clustering_features) > 5:
            try:
                # Preprocess data
                df_processed = preprocess_data(df[clustering_features].copy())
                df_processed = df_processed.select_dtypes(include=[np.number])

                # CRITICAL FIX: Force all columns to numeric
                for col in df_processed.columns:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

                # Fill any NaN values
                df_processed = df_processed.fillna(df_processed.median())

                # Verify all columns are numeric
                if not all(df_processed.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                    st.error("Some columns are still not numeric. Please check your data.")
                    st.stop()

                # Scale data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_processed)

                # Clustering
                if clustering_method == "K-Means":
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                else:
                    model = GaussianMixture(n_components=n_clusters, random_state=42)

                clusters = model.fit_predict(X_scaled)

                # Add clusters to PROCESSED dataframe (not original df)
                df_processed['Cluster'] = clusters

                # Calculate metrics
                silhouette = silhouette_score(X_scaled, clusters)
                davies_bouldin = davies_bouldin_score(X_scaled, clusters)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Silhouette Score", f"{silhouette:.3f}")
                with col2:
                    st.metric("Davies-Bouldin Score", f"{davies_bouldin:.3f}")
                with col3:
                    st.metric("Number of Clusters", n_clusters)

                st.markdown("---")

                # PCA for visualization
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                df_processed['PCA1'] = X_pca[:, 0]
                df_processed['PCA2'] = X_pca[:, 1]

                st.subheader("Customer Segments Visualization")
                fig = px.scatter(df_processed, x='PCA1', y='PCA2', color='Cluster',
                               title=f"{clustering_method} Clustering",
                               color_continuous_scale='viridis')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Cluster Size Distribution")
                    cluster_counts = df_processed['Cluster'].value_counts().sort_index()
                    fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                               labels={'x': 'Cluster', 'y': 'Count'},
                               color=cluster_counts.values, color_continuous_scale='blues',
                               text=cluster_counts.values)
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Cluster Characteristics")
                    # CRITICAL FIX: Use ONLY the processed dataframe columns
                    # Remove 'Cluster', 'PCA1', 'PCA2' from the list
                    numeric_cols = [col for col in df_processed.columns 
                                   if col not in ['Cluster', 'PCA1', 'PCA2']][:5]

                    if len(numeric_cols) > 0:
                        # Group by cluster using PROCESSED dataframe
                        cluster_profile = df_processed.groupby('Cluster')[numeric_cols].mean()
                        st.dataframe(cluster_profile.style.background_gradient(cmap='RdYlGn'), 
                                   use_container_width=True)
                    else:
                        st.info("No numeric columns available for profiling.")

                # Detailed cluster insights
                st.subheader("💡 Cluster Insights")
                for cluster_id in range(n_clusters):
                    with st.expander(f"Cluster {cluster_id} - {len(df_processed[df_processed['Cluster']==cluster_id])} customers"):
                        cluster_data = df_processed[df_processed['Cluster'] == cluster_id]

                        # Show statistics for numeric columns only
                        cols_to_show = [col for col in numeric_cols if col in cluster_data.columns]

                        if len(cols_to_show) > 0:
                            stats_df = cluster_data[cols_to_show].describe().T
                            st.dataframe(stats_df[['mean', 'std', 'min', 'max']], use_container_width=True)
                        else:
                            st.info("No data available for this cluster.")

                # Download results
                download_df = df_processed[['Cluster', 'PCA1', 'PCA2'] + numeric_cols[:5]]
                st.download_button(
                    label="📥 Download Clustering Results",
                    data=download_df.to_csv(index=False),
                    file_name="clustering_results.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error in clustering: {str(e)}")
                st.info("Try adjusting the number of clusters or check your data.")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("Not enough features available for clustering.")

    # ========================================================================
    # TAB 4: ASSOCIATION RULES
    # ========================================================================
    with tab4:
        st.header("🔗 Association Rule Mining")

        with st.sidebar:
            st.markdown("### 🔗 Association Rules")
            min_support = st.slider("Min Support (%)", 1, 50, 10, key="support") / 100
            min_confidence = st.slider("Min Confidence (%)", 10, 100, 70, key="confidence") / 100
            top_n_rules = st.slider("Top N Rules", 5, 50, 10)

        if 'Q13_Game_Types_Preferred' in df.columns and 'Q23_Leisure_Venues_Visited' in df.columns:
            try:
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
                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

                    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

                    if len(frequent_itemsets) > 0:
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

                        if len(rules) > 0:
                            rules = rules.sort_values('confidence', ascending=False).head(top_n_rules)

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Frequent Itemsets", len(frequent_itemsets))
                            with col2:
                                st.metric("Association Rules", len(rules))
                            with col3:
                                st.metric("Avg Confidence", f"{rules['confidence'].mean():.2%}")

                            st.markdown("---")
                            st.subheader(f"Top {len(rules)} Association Rules")

                            # Convert frozensets to strings
                            rules_display = rules.copy()
                            rules_display['antecedents'] = rules_display['antecedents'].apply(
                                lambda x: ', '.join(list(x)) if isinstance(x, frozenset) else str(x)
                            )
                            rules_display['consequents'] = rules_display['consequents'].apply(
                                lambda x: ', '.join(list(x)) if isinstance(x, frozenset) else str(x)
                            )
                            rules_display['support'] = rules_display['support'].apply(lambda x: f"{x:.1%}")
                            rules_display['confidence'] = rules_display['confidence'].apply(lambda x: f"{x:.1%}")
                            rules_display['lift'] = rules_display['lift'].apply(lambda x: f"{x:.2f}")

                            st.dataframe(rules_display[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
                                       use_container_width=True)

                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("Support vs Confidence")
                                fig = px.scatter(rules, x='support', y='confidence', size='lift',
                                               color='lift', color_continuous_scale='viridis')
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                st.subheader("Lift Distribution")
                                fig = px.histogram(rules, x='lift', nbins=20, 
                                                 color_discrete_sequence=['#667eea'])
                                st.plotly_chart(fig, use_container_width=True)

                            st.download_button(
                                label="📥 Download Association Rules",
                                data=rules_display.to_csv(index=False),
                                file_name="association_rules.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning(f"No rules found with confidence ≥ {min_confidence:.0%}.")
                    else:
                        st.warning(f"No frequent itemsets found with support ≥ {min_support:.0%}.")
                else:
                    st.error("No valid transactions found.")
            except Exception as e:
                st.error(f"Error in association rules: {str(e)}")
        else:
            st.error("Required columns not found.")

    # ========================================================================
    # TAB 5: REGRESSION
    # ========================================================================
    with tab5:
        st.header("💰 Regression Analysis")
        st.markdown("### Linear, Ridge, and Lasso Regression")

        with st.sidebar:
            st.markdown("### 💰 Regression Settings")
            test_size_reg = st.slider("Test Size (%)", 10, 40, 20, key="test_reg") / 100
            selected_models_reg = st.multiselect(
                "Select Models",
                ["Linear Regression", "Ridge", "Lasso", "Decision Tree", "Random Forest", "Gradient Boosting"],
                default=["Linear Regression", "Ridge", "Lasso"]
            )

        target_col = 'Q37_Total_WTP_Per_Visit_AED'

        if target_col in df.columns and len(selected_models_reg) > 0:
            try:
                predictor_features = [
                    'Q1_Age', 'Q6_Monthly_Income_AED', 'Q11_Play_Video_Games',
                    'Q15_Hours_Per_Week', 'Q38_Price_Sensitivity',
                    'Q26_Food_Quality_Importance', 'Q45_Interest_In_Concept',
                    'Q47_Expected_Visit_Frequency', 'Q21_Social_Aspect_Importance'
                ]

                predictor_features = [f for f in predictor_features if f in df.columns]

                if len(predictor_features) > 3:
                    spending_mapping = {
                        '50-100 AED': 75, '101-150 AED': 125, '151-200 AED': 175,
                        '201-300 AED': 250, '301-400 AED': 350, 'Above 400 AED': 450
                    }

                    df_reg = df.copy()
                    df_reg[target_col] = df_reg[target_col].map(spending_mapping)

                    df_processed = preprocess_data(df_reg[predictor_features + [target_col]])
                    df_processed = df_processed.select_dtypes(include=[np.number])

                    X = df_processed[predictor_features]
                    y = df_processed[target_col]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_reg, random_state=42)

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    models_dict = {
                        "Linear Regression": LinearRegression(),
                        "Ridge": Ridge(alpha=1.0),
                        "Lasso": Lasso(alpha=1.0),
                        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
                        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
                        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
                    }

                    results = {}

                    for name in selected_models_reg:
                        if name in models_dict:
                            model = models_dict[name]

                            if name in ["Linear Regression", "Ridge", "Lasso"]:
                                model.fit(X_train_scaled, y_train)
                                y_pred = model.predict(X_test_scaled)
                            else:
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)

                            r2 = r2_score(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            mae = mean_absolute_error(y_test, y_pred)

                            results[name] = {
                                'R² Score': r2,
                                'RMSE': rmse,
                                'MAE': mae,
                                'predictions': y_pred
                            }

                    st.subheader("📊 Model Performance")

                    comparison_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'R² Score': [results[m]['R² Score'] for m in results.keys()],
                        'RMSE (AED)': [results[m]['RMSE'] for m in results.keys()],
                        'MAE (AED)': [results[m]['MAE'] for m in results.keys()]
                    })

                    st.dataframe(comparison_df.style.background_gradient(subset=['R² Score'], cmap='RdYlGn')
                                                    .background_gradient(subset=['RMSE (AED)'], cmap='RdYlGn_r')
                                                    .format({'R² Score': '{:.3f}', 'RMSE (AED)': '{:.2f}', 
                                                            'MAE (AED)': '{:.2f}'}),
                               use_container_width=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("R² Score Comparison")
                        fig = px.bar(comparison_df, x='Model', y='R² Score',
                                   color='R² Score', color_continuous_scale='viridis',
                                   text='R² Score')
                        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("RMSE Comparison")
                        fig = px.bar(comparison_df, x='Model', y='RMSE (AED)',
                                   color='RMSE (AED)', color_continuous_scale='reds',
                                   text='RMSE (AED)')
                        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)

                    best_model_name = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']
                    st.success(f"🏆 Best Model: **{best_model_name}** (R² = {results[best_model_name]['R² Score']:.3f})")

                    st.subheader(f"{best_model_name}: Predicted vs Actual")
                    pred_actual_df = pd.DataFrame({
                        'Actual': y_test,
                        'Predicted': results[best_model_name]['predictions']
                    })

                    fig = px.scatter(pred_actual_df, x='Actual', y='Predicted')
                    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                           y=[y_test.min(), y_test.max()],
                                           mode='lines', name='Perfect Prediction',
                                           line=dict(dash='dash', color='red')))
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    st.download_button(
                        label="📥 Download Results",
                        data=comparison_df.to_csv(index=False),
                        file_name="regression_results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Not enough features.")
            except Exception as e:
                st.error(f"Error in regression: {str(e)}")
        else:
            if len(selected_models_reg) == 0:
                st.info("Please select at least one regression model.")
            else:
                st.error("Target variable not found.")

    # ========================================================================
    # TAB 6: DYNAMIC PRICING
    # ========================================================================
    with tab6:
        st.header("🎛️ Dynamic Pricing Engine")

        with st.sidebar:
            st.markdown("### 🎛️ Pricing Parameters")
            base_price = st.number_input("Base Price (AED)", 50, 500, 150, step=10)
            max_discount = st.slider("Max Discount (%)", 0, 50, 20) / 100

        required_cols = ['Q17_Gaming_Cafe_Visits_Past_12mo', 'Q47_Expected_Visit_Frequency', 'Q45_Interest_In_Concept']

        if all(col in df.columns for col in required_cols):
            try:
                df_price = preprocess_data(df[required_cols].copy())
                df_price = df_price.select_dtypes(include=[np.number])

                df_price['Loyalty_Score'] = (
                    df_price[required_cols[0]] * 30 +
                    df_price[required_cols[1]] * 25 +
                    df_price[required_cols[2]] * 20
                ).clip(0, 100)

                df_price['Loyalty_Tier'] = pd.cut(df_price['Loyalty_Score'],
                                                  bins=[0, 40, 60, 80, 100],
                                                  labels=['Bronze', 'Silver', 'Gold', 'Platinum'])

                df_price['Loyalty_Discount'] = (df_price['Loyalty_Score'] / 100) * max_discount
                df_price['Dynamic_Price'] = base_price * (1 - df_price['Loyalty_Discount'])
                df_price['Savings'] = base_price - df_price['Dynamic_Price']
                df_price['Discount_Pct'] = (df_price['Savings'] / base_price) * 100

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Base Price", f"{base_price} AED")
                with col2:
                    st.metric("Avg Price", f"{df_price['Dynamic_Price'].mean():.2f} AED")
                with col3:
                    st.metric("Avg Discount", f"{df_price['Discount_Pct'].mean():.1f}%")
                with col4:
                    st.metric("Revenue Potential", f"{df_price['Dynamic_Price'].sum():,.0f} AED")

                st.markdown("---")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Price by Loyalty Tier")
                    fig = px.box(df_price, x='Loyalty_Tier', y='Dynamic_Price', color='Loyalty_Tier',
                               color_discrete_map={'Bronze': '#CD7F32', 'Silver': '#C0C0C0',
                                                  'Gold': '#FFD700', 'Platinum': '#E5E4E2'})
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Loyalty Score Distribution")
                    fig = px.histogram(df_price, x='Loyalty_Score', nbins=30,
                                     color_discrete_sequence=['#667eea'])
                    st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    label="📥 Download Rate Card",
                    data=df_price.to_csv(index=False),
                    file_name="digital_rate_card.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error in pricing: {str(e)}")
        else:
            st.warning("Required columns not found.")

else:
    st.warning("⚠️ Please upload data.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white;'>
    <p>🎮 Gaming Cafe Analytics Dashboard | Clustering Error Fixed ✅</p>
</div>
""", unsafe_allow_html=True)
