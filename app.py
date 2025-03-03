import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Crime Data Analysis Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved design
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        background: linear-gradient(to right, #e0e8ff, #f0f2f6);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #4e7496;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50;
    }
    h1, h2, h3 {
        color: #1e3d59;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chart-container {
        background-color: transparent;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🔍 Crime Data Analysis Dashboard")
st.markdown("""
This interactive dashboard provides insights into Austin crime data, including trends, patterns, and predictive analytics.
Explore the data through various visualizations and analysis tools.
""")

# Load data with caching
@st.cache_data
def load_austin_crime_data():
    try:
        file_path = "Crime_Reports.csv"  # Replace with your actual file path
        df = pd.read_csv(file_path)
        st.success("Successfully loaded Austin crime data")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.warning("Using sample data for demonstration as fallback.")
        return generate_sample_data()

def generate_sample_data():
    df = pd.DataFrame({
        'Incident Number': range(1000, 2000),
        'Highest Offense Description': np.random.choice(['THEFT', 'ASSAULT', 'BURGLARY', 'ROBBERY', 'AUTO THEFT', 'FAMILY DISTURBANCE', 'CRIMINAL MISCHIEF', 'DRUG POSSESSION'], 1000),
        'Highest Offense Code': np.random.randint(100, 999, 1000),
        'Family Violence': np.random.choice(['Y', 'N'], 1000),
        'Occurred Date Time': [datetime.now() - timedelta(days=np.random.randint(1, 730)) for _ in range(1000)],
        'Report Date Time': [datetime.now() - timedelta(days=np.random.randint(1, 730)) for _ in range(1000)],
        'Location Type': np.random.choice(['RESIDENCE', 'STREET', 'COMMERCIAL', 'PARKING LOT', 'APARTMENT', 'BUSINESS', 'PARK'], 1000),
        'APD Sector': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'], 1000),
        'APD District': np.random.choice(['DAVID', 'EDWARD', 'FRANK', 'GEORGE', 'HENRY', 'IDA', 'JOHN', 'KING', 'LINCOLN'], 1000),
        'Council District': np.random.randint(1, 11, 1000),
        'Clearance Status': np.random.choice(['CLEARED BY ARREST', 'NOT CLEARED', 'EXCEPTIONALLY CLEARED', 'SUSPENDED', 'UNFOUNDED'], 1000),
        'Clearance Date': [datetime.now() - timedelta(days=np.random.randint(1, 600)) for _ in range(1000)],
        'UCR Category': np.random.choice(['Property Crime', 'Violent Crime', 'Quality of Life', 'Other'], 1000),
        'Category Description': np.random.choice(['THEFT', 'ASSAULT', 'BURGLARY', 'ROBBERY', 'AUTO THEFT', 'FAMILY DISTURBANCE'], 1000),
        'X Coordinate': np.random.uniform(3110000, 3160000, 1000),
        'Y Coordinate': np.random.uniform(10060000, 10110000, 1000),
        'Latitude': np.random.uniform(30.1, 30.5, 1000),
        'Longitude': np.random.uniform(-97.9, -97.5, 1000),
    })
    return df

# Load and preprocess data
df = load_austin_crime_data()
st.session_state.data_loaded = True

# Process dates
date_columns = ['Occurred Date Time', 'Report Date Time', 'Clearance Date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Extract date and time components
if 'Occurred Date Time' in df.columns:
    df['Occurred Date'] = df['Occurred Date Time'].dt.date
    df['Occurred Time'] = df['Occurred Date Time'].dt.time
if 'Report Date Time' in df.columns:
    df['Report Date'] = df['Report Date Time'].dt.date
    df['Report Time'] = df['Report Date Time'].dt.time

# Sidebar for filters
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/police-badge.png", width=80)
    st.header("Dashboard Controls")
    st.info(f"Analyzing {len(df)} crime incidents")
    st.markdown("---")
    st.markdown("### Filters")

    # Collect all filters at once for efficiency
    filters = {}
    if 'Occurred Date' in df.columns:
        min_date = pd.to_datetime(df['Occurred Date Time']).min().date()
        max_date = pd.to_datetime(df['Occurred Date Time']).max().date()
        filters['date_range'] = st.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    if 'Highest Offense Description' in df.columns:
        offense_options = ['All'] + sorted(df['Highest Offense Description'].unique().tolist())
        filters['offense'] = st.selectbox("Offense Type", offense_options)
    
    if 'Council District' in df.columns:
        district_options = ['All'] + sorted([str(d) for d in df['Council District'].unique().tolist()])
        filters['district'] = st.selectbox("Council District", district_options)
    
    if 'Clearance Status' in df.columns:
        status_options = ['All'] + sorted([s for s in df['Clearance Status'].unique().tolist() if isinstance(s, str)])
        filters['status'] = st.selectbox("Clearance Status", status_options)

    # Apply all filters at once
    filtered_df = df.copy()
    if filters.get('date_range') and len(filters['date_range']) == 2:
        start_date, end_date = filters['date_range']
        mask = (pd.to_datetime(filtered_df['Occurred Date Time']).dt.date >= start_date) & \
               (pd.to_datetime(filtered_df['Occurred Date Time']).dt.date <= end_date)
        filtered_df = filtered_df.loc[mask]
    if filters.get('offense') and filters['offense'] != 'All':
        filtered_df = filtered_df[filtered_df['Highest Offense Description'] == filters['offense']]
    if filters.get('district') and filters['district'] != 'All':
        filtered_df = filtered_df[filtered_df['Council District'].astype(str) == filters['district']]
    if filters.get('status') and filters['status'] != 'All':
        filtered_df = filtered_df[filtered_df['Clearance Status'] == filters['status']]

    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This dashboard analyzes Austin crime data to identify patterns, trends, and insights
    that can help in understanding and addressing crime in the community.
    """)

# Main content with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔎 Detailed Analysis", "🗺️ Spatial Patterns", "🤖 Predictive Model"])

with tab1:
    st.header("Crime Data Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Incidents", len(filtered_df))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Clearance Status' in filtered_df.columns:
            cleared_statuses = ['CLEARED BY ARREST', 'EXCEPTIONALLY CLEARED']
            cleared_count = filtered_df[filtered_df['Clearance Status'].isin(cleared_statuses)].shape[0]
            total_valid = filtered_df['Clearance Status'].notna().sum()
            clearance_rate = round((cleared_count / total_valid) * 100, 1) if total_valid > 0 else 0
            st.metric("Clearance Rate", f"{clearance_rate}%")
        else:
            st.metric("Clearance Rate", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Family Violence' in filtered_df.columns:
            fv_count = filtered_df[filtered_df['Family Violence'] == 'Y'].shape[0]
            fv_rate = round((fv_count / len(filtered_df)) * 100, 1) if len(filtered_df) > 0 else 0
            st.metric("Family Violence", f"{fv_rate}%")
        else:
            st.metric("Family Violence", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'Occurred Date Time' in filtered_df.columns:
            recent_crimes = filtered_df[filtered_df['Occurred Date Time'] >= (datetime.now() - timedelta(days=30))].shape[0]
            st.metric("Last 30 Days", recent_crimes)
        else:
            st.metric("Last 30 Days", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Crime Incidents Over Time")
        if 'Occurred Date Time' in filtered_df.columns:
            filtered_df['Month'] = filtered_df['Occurred Date Time'].dt.to_period('M')
            time_series = filtered_df.groupby('Month').size().reset_index(name='Count')
            time_series['Month'] = time_series['Month'].astype(str)
            fig = px.line(time_series, x='Month', y='Count', markers=True, line_shape='spline', color_discrete_sequence=['#1e3d59'])
            fig.update_layout(xaxis_title="Month", yaxis_title="Number of Incidents", plot_bgcolor='rgba(255,255,255,0.8)', height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Date information not available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Top 10 Offense Types")
        if 'Highest Offense Description' in filtered_df.columns:
            offense_counts = filtered_df['Highest Offense Description'].value_counts().head(10).reset_index()
            offense_counts.columns = ['Offense', 'Count']
            fig = px.bar(offense_counts, x='Count', y='Offense', orientation='h', color='Count', color_continuous_scale='Blues')
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Number of Incidents", yaxis_title="", plot_bgcolor='rgba(255,255,255,0.8)', height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Offense information not available")
        st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Clearance Status Distribution")
        if 'Clearance Status' in filtered_df.columns:
            clearance_counts = filtered_df['Clearance Status'].value_counts().reset_index()
            clearance_counts.columns = ['Status', 'Count']
            fig = px.pie(clearance_counts, names='Status', values='Count', hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(legend_title="Clearance Status", plot_bgcolor='rgba(0,0,0,0)', height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Clearance status information not available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Family Violence Incidents")
        if 'Family Violence' in filtered_df.columns:
            fv_counts = filtered_df['Family Violence'].value_counts().reset_index()
            fv_counts.columns = ['Family Violence', 'Count']
            fv_counts['Family Violence'] = fv_counts['Family Violence'].map({'Y': 'Yes', 'N': 'No'})
            fig = px.pie(fv_counts, names='Family Violence', values='Count', hole=0.4, color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
            fig.update_layout(legend_title="Family Violence", plot_bgcolor='rgba(0,0,0,0)', height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Family violence information not available")
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.header("Detailed Crime Analysis")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Crime by Time of Day")
    if 'Occurred Time' in filtered_df.columns:
        filtered_df['Hour'] = pd.to_datetime(filtered_df['Occurred Time'], format='%H:%M:%S', errors='coerce').dt.hour
        hour_counts = filtered_df.groupby('Hour').size().reset_index(name='Count')
        fig = px.bar(hour_counts, x='Hour', y='Count', color='Count', color_continuous_scale='Viridis')
        fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1), xaxis_title="Hour of Day (24-hour)", yaxis_title="Number of Incidents", plot_bgcolor='rgba(255,255,255,0.8)', height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Time information not available")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Crimes by Day of Week")
        if 'Occurred Date Time' in filtered_df.columns:
            filtered_df['Day of Week'] = filtered_df['Occurred Date Time'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_counts = filtered_df['Day of Week'].value_counts().reindex(day_order).reset_index()
            day_counts.columns = ['Day', 'Count']
            fig = px.bar(day_counts, x='Day', y='Count', color='Count', color_continuous_scale='Blues')
            fig.update_layout(xaxis_title="", yaxis_title="Number of Incidents", plot_bgcolor='rgba(255,255,255,0.8)', height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Date information not available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Crimes by Month")
        if 'Occurred Date Time' in filtered_df.columns:
            filtered_df['Month'] = filtered_df['Occurred Date Time'].dt.month_name()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            month_counts = filtered_df['Month'].value_counts().reindex(month_order).reset_index()
            month_counts.columns = ['Month', 'Count']
            fig = px.bar(month_counts, x='Month', y='Count', color='Count', color_continuous_scale='Blues')
            fig.update_layout(xaxis_title="", yaxis_title="Number of Incidents", plot_bgcolor='rgba(255,255,255,0.8)', height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Date information not available")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Offense Code and Council District Relationship")
    col1, col2 = st.columns([3, 1])
    with col1:
        if 'Highest Offense Code' in filtered_df.columns and 'Council District' in filtered_df.columns:
            if filtered_df['Council District'].dtype == 'object':
                filtered_df['Council District'] = pd.to_numeric(filtered_df['Council District'], errors='coerce')
            fig = px.scatter(filtered_df, x='Council District', y='Highest Offense Code', color='Clearance Status', size_max=10, opacity=0.7, color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_layout(xaxis_title="Council District", yaxis_title="Offense Code", plot_bgcolor='rgba(255,255,255,0.8)', height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("District or offense code information not available")
    with col2:
        if 'Council District' in filtered_df.columns:
            st.subheader("District Count")
            district_counts = filtered_df['Council District'].value_counts().sort_index().reset_index()
            district_counts.columns = ['District', 'Count']
            fig = px.bar(district_counts, x='District', y='Count', color='Count', color_continuous_scale='Blues')
            fig.update_layout(xaxis_title="", yaxis_title="", plot_bgcolor='rgba(255,255,255,0.8)', height=400)
            st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Data Explorer")
    if st.checkbox("Show filtered data sample"):
        st.write(filtered_df.head(100))
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.header("Spatial Crime Patterns")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Crime by Council District")
    if 'Council District' in filtered_df.columns:
        district_crime = filtered_df.groupby('Council District').size().reset_index(name='Count')
        fig = px.bar(district_crime, x='Council District', y='Count', color='Count', color_continuous_scale='Viridis', labels={'Council District': 'District', 'Count': 'Number of Incidents'})
        fig.update_layout(xaxis_title="Council District", yaxis_title="Number of Incidents", plot_bgcolor='rgba(255,255,255,0.8)', height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Council district information not available")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Crimes by Location Type")
        if 'Location Type' in filtered_df.columns:
            location_counts = filtered_df['Location Type'].value_counts().head(10).reset_index()
            location_counts.columns = ['Location Type', 'Count']
            fig = px.pie(location_counts, names='Location Type', values='Count', color_discrete_sequence=px.colors.sequential.Plasma_r)
            fig.update_layout(legend_title="Location Type", plot_bgcolor='rgba(0,0,0,0)', height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Location type information not available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("APD Sector Distribution")
        if 'APD Sector' in filtered_df.columns:
            sector_counts = filtered_df['APD Sector'].value_counts().reset_index()
            sector_counts.columns = ['APD Sector', 'Count']
            fig = px.bar(sector_counts, x='APD Sector', y='Count', color='Count', color_continuous_scale='Viridis')
            fig.update_layout(xaxis_title="APD Sector", yaxis_title="Number of Incidents", plot_bgcolor='rgba(255,255,255,0.8)', height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("APD Sector information not available")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Crime Locations Map")
    if 'Latitude' in filtered_df.columns and 'Longitude' in filtered_df.columns:
        map_data = filtered_df.dropna(subset=['Latitude', 'Longitude']).sample(min(1000, len(filtered_df)))
        fig = px.scatter_mapbox(map_data, lat='Latitude', lon='Longitude', color='Highest Offense Description', size_max=10, zoom=10, mapbox_style="open-street-map", opacity=0.7, height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Coordinate information not available for map visualization")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Offense Types by Location")
    if 'Highest Offense Description' in filtered_df.columns and 'Location Type' in filtered_df.columns:
        top_offenses = filtered_df['Highest Offense Description'].value_counts().head(5).index.tolist()
        top_locations = filtered_df['Location Type'].value_counts().head(5).index.tolist()
        filtered_subset = filtered_df[(filtered_df['Highest Offense Description'].isin(top_offenses)) & (filtered_df['Location Type'].isin(top_locations))]
        cross_tab = pd.crosstab(filtered_subset['Highest Offense Description'], filtered_subset['Location Type']).reset_index()
        cross_tab_melted = pd.melt(cross_tab, id_vars=['Highest Offense Description'], var_name='Location Type', value_name='Count')
        fig = px.density_heatmap(cross_tab_melted, x='Location Type', y='Highest Offense Description', z='Count', color_continuous_scale='Viridis')
        fig.update_layout(xaxis_title="Location Type", yaxis_title="Offense Type", plot_bgcolor='rgba(255,255,255,0.8)', height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Offense or location information not available")
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.header("Predictive Analytics")
    st.warning("This is a simplified model for demonstration purposes.")
    if st.button("Train Clearance Status Prediction Model"):
        with st.spinner('Processing data and training model...'):
            model_df = df.copy()
            required_cols = ['Clearance Status', 'Highest Offense Code', 'Council District']
            missing_cols = [col for col in required_cols if col not in model_df.columns]
            if missing_cols:
                st.error(f"Missing required columns for modeling: {', '.join(missing_cols)}")
            else:
                # Drop datetime and irrelevant columns
                drop_cols = ['Incident Number', 'Occurred Date Time', 'Report Date Time', 'Clearance Date', 'Occurred Date', 'Occurred Time', 'Report Date', 'Report Time']
                model_cols = [col for col in model_df.columns if col not in drop_cols]
                model_df = model_df[model_cols].copy()
                
                # Sample data for speed
                if len(model_df) > 10000:
                    model_df = model_df.sample(10000, random_state=42)
                
                # Define column types
                categorical_cols = model_df.select_dtypes(include='object').columns
                numerical_cols = model_df.select_dtypes(include=['int64', 'float64']).columns
                
                # Impute missing values
                cat_imputer = SimpleImputer(strategy='most_frequent')
                num_imputer = SimpleImputer(strategy='median')
                model_df[categorical_cols] = cat_imputer.fit_transform(model_df[categorical_cols])
                model_df[numerical_cols] = num_imputer.fit_transform(model_df[numerical_cols])
                
                # Encode variables
                encoder = OrdinalEncoder()
                le = LabelEncoder()
                model_df['Clearance Status'] = le.fit_transform(model_df['Clearance Status'])
                cat_cols_to_encode = [col for col in categorical_cols if col != 'Clearance Status']
                if cat_cols_to_encode:
                    model_df[cat_cols_to_encode] = encoder.fit_transform(model_df[cat_cols_to_encode])
                
                # Split data
                X = model_df.drop('Clearance Status', axis=1)
                y = model_df['Clearance Status']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale numerical features
                numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
                scaler = StandardScaler()
                X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
                X_test[numeric_features] = scaler.transform(X_test[numeric_features])
                
                # Train model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.success(f"Model training completed with accuracy: {accuracy:.2f}")
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.subheader("Feature Importance")
                    fig = px.bar(feature_importance.head(10), x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Importance", yaxis_title="Feature", plot_bgcolor='rgba(255,255,255,0.8)', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Confusion matrix
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.subheader("Confusion Matrix")
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_names = le.classes_
                fig = px.imshow(conf_matrix, x=class_names, y=class_names, color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual", color="Count"))
                fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", plot_bgcolor='rgba(255,255,255,0.8)', height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Classification report
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0))
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Prediction section
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.subheader("Predict Clearance Status")
                st.write("Predict clearance status for hypothetical cases.")
                input_data = {}
                top_features = feature_importance.head(5)['Feature'].tolist()
                col1, col2 = st.columns(2)
                with col1:
                    for feature in top_features:
                        if feature in categorical_cols:
                            original_values = df[feature].dropna().unique()
                            input_data[feature] = st.selectbox(f"Select {feature}", options=original_values)
                        else:
                            min_val = float(df[feature].min())
                            max_val = float(df[feature].max())
                            default_val = float(df[feature].median())
                            input_data[feature] = st.number_input(f"Enter {feature}", min_value=min_val, max_value=max_val, value=default_val)
                with col2:
                    st.markdown("### Prediction Result")
                    if st.button("Predict"):
                        input_df = pd.DataFrame([input_data])
                        for col in X.columns:
                            if col not in input_df.columns:
                                if col in numerical_cols:
                                    input_df[col] = df[col].median()
                                else:
                                    input_df[col] = df[col].mode()[0]
                        input_df = input_df[X.columns]
                        if len(numeric_features) > 0:
                            input_df[numeric_features] = scaler.transform(input_df[numeric_features])
                        prediction = model.predict(input_df)
                        prediction_proba = model.predict_proba(input_df)
                        predicted_status = le.inverse_transform(prediction)[0]
                        st.success(f"Predicted Clearance Status: {predicted_status}")
                        proba_df = pd.DataFrame({'Status': le.classes_, 'Probability': prediction_proba[0]}).sort_values('Probability', ascending=False)
                        fig = px.bar(proba_df, x='Status', y='Probability', color='Probability', color_continuous_scale='Blues')
                        fig.update_layout(xaxis_title="Clearance Status", yaxis_title="Probability", plot_bgcolor='rgba(255,255,255,0.8)', height=300)
                        st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Key Insights and Recommendations")
    st.write("""
    1. **Temporal Patterns**: Optimize patrol schedules based on time, day, and month trends.
    2. **Spatial Distribution**: Target high-crime districts and location types.
    3. **Offense Patterns**: Allocate resources to prevalent crime types.
    4. **Clearance Rates**: Enhance investigation strategies using predictive insights.
    5. **Family Violence**: Develop support programs based on prevalence data.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #1e3d59; color: white; border-radius: 10px;">
    <p>Crime Data Analysis Dashboard</p>
    <p>For demonstration and educational purposes only.</p>
</div>
""", unsafe_allow_html=True)

# Downloadable reports
if st.sidebar.checkbox("Generate Downloadable Report"):
    st.sidebar.markdown("### Download Options")
    report_type = st.sidebar.selectbox("Select Report Type", ["Summary Report", "Detailed Analysis", "District Analysis"])
    if st.sidebar.button("Generate Report"):
        st.sidebar.success("Report generated successfully!")
        if report_type == "Summary Report":
            report_data = f"""
            # Crime Data Analysis Summary Report
            ## Overview
            - Total Incidents: {len(filtered_df)}
            - Date Range: {filtered_df['Occurred Date Time'].min().date()} to {filtered_df['Occurred Date Time'].max().date()}
            - Most Common Offense: {filtered_df['Highest Offense Description'].value_counts().index[0]}
            """
        elif report_type == "Detailed Analysis":
            report_data = f"""
            # Detailed Crime Analysis Report
            ## Temporal Analysis
            - Peak Crime Hour: {filtered_df.groupby(filtered_df['Occurred Date Time'].dt.hour).size().idxmax()}:00
            - Most Active Day: {filtered_df['Occurred Date Time'].dt.day_name().value_counts().index[0]}
            - Most Active Month: {filtered_df['Occurred Date Time'].dt.month_name().value_counts().index[0]}
            """
        else:
            report_data = f"""
            # District-Based Crime Analysis
            ## Council District Analysis
            - Highest Crime District: {filtered_df['Council District'].value_counts().index[0]}
            - Most Common Offense in District {filtered_df['Council District'].value_counts().index[0]}: {filtered_df[filtered_df['Council District'] == filtered_df['Council District'].value_counts().index[0]]['Highest Offense Description'].value_counts().index[0]}
            """
        st.sidebar.download_button(label="Download Report", data=report_data, file_name=f"{report_type.lower().replace(' ', '_')}.md", mime="text/markdown")