"""
Customer Segmentation Page
Advanced ML-powered customer clustering and segment analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="👥",
    layout="wide"
)

class CustomerSegmentationEngine:
    """Advanced Customer Segmentation Engine with multiple ML algorithms"""

    def __init__(self):
        self.data = None
        self.scaled_data = None
        self.scaler = None
        self.clusters = None
        self.cluster_labels = None
        self.n_clusters = 5
        self.algorithm = 'kmeans'
        self.features = []

    def load_data(self, data):
        """Load and prepare data for segmentation"""
        self.data = data.copy()
        return True

    def preprocess_data(self, features, scaling_method='standard'):
        """Preprocess data for clustering"""
        try:
            # Select features
            self.features = features
            feature_data = self.data[features].copy()

            # Handle missing values
            feature_data = feature_data.fillna(feature_data.median())

            # Scale data
            if scaling_method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = MinMaxScaler()

            self.scaled_data = self.scaler.fit_transform(feature_data)

            return True, "Data preprocessed successfully"

        except Exception as e:
            return False, f"Error preprocessing data: {str(e)}"

    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using multiple metrics"""
        if self.scaled_data is None:
            return None

        metrics = {
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }

        cluster_range = range(2, max_clusters + 1)

        for n in cluster_range:
            # KMeans clustering
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.scaled_data)

            # Calculate metrics
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(self.scaled_data, labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(self.scaled_data, labels))
            metrics['davies_bouldin'].append(davies_bouldin_score(self.scaled_data, labels))

        return cluster_range, metrics

    def perform_clustering(self, algorithm='kmeans', n_clusters=5, **kwargs):
        """Perform clustering using specified algorithm"""
        if self.scaled_data is None:
            return False, "Data not preprocessed"

        self.algorithm = algorithm
        self.n_clusters = n_clusters

        try:
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

            elif algorithm == 'gaussian_mixture':
                model = GaussianMixture(n_components=n_clusters, random_state=42)

            elif algorithm == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)

            elif algorithm == 'dbscan':
                eps = kwargs.get('eps', 0.5)
                min_samples = kwargs.get('min_samples', 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)

            else:
                return False, f"Unknown algorithm: {algorithm}"

            # Fit model and get labels
            if algorithm == 'gaussian_mixture':
                model.fit(self.scaled_data)
                self.cluster_labels = model.predict(self.scaled_data)
            else:
                self.cluster_labels = model.fit_predict(self.scaled_data)

            # Add cluster labels to original data
            self.data['Cluster'] = self.cluster_labels

            return True, f"Clustering completed with {algorithm}"

        except Exception as e:
            return False, f"Error in clustering: {str(e)}"

    def get_cluster_profiles(self):
        """Generate detailed cluster profiles"""
        if self.cluster_labels is None:
            return None

        profiles = {}

        for cluster_id in np.unique(self.cluster_labels):
            if cluster_id == -1:  # DBSCAN noise points
                continue

            cluster_data = self.data[self.data['Cluster'] == cluster_id]

            profile = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.data) * 100,
                'statistics': {}
            }

            # Calculate statistics for each feature
            for feature in self.features:
                if feature in cluster_data.columns:
                    profile['statistics'][feature] = {
                        'mean': cluster_data[feature].mean(),
                        'median': cluster_data[feature].median(),
                        'std': cluster_data[feature].std(),
                        'min': cluster_data[feature].min(),
                        'max': cluster_data[feature].max()
                    }

            profiles[f'Cluster_{cluster_id}'] = profile

        return profiles

    def get_cluster_insights(self):
        """Generate business insights for each cluster"""
        profiles = self.get_cluster_profiles()
        if not profiles:
            return None

        insights = {}

        for cluster_name, profile in profiles.items():
            cluster_insights = []
            stats = profile['statistics']

            # Generate insights based on feature values
            if 'credit_score' in stats:
                score = stats['credit_score']['mean']
                if score >= 750:
                    cluster_insights.append("Excellent credit profile - Premium customers")
                elif score >= 650:
                    cluster_insights.append("Good credit profile - Standard customers")
                else:
                    cluster_insights.append("Poor credit profile - High-risk customers")

            if 'income' in stats:
                income = stats['income']['mean']
                if income >= 100000:
                    cluster_insights.append("High-income segment")
                elif income >= 50000:
                    cluster_insights.append("Middle-income segment")
                else:
                    cluster_insights.append("Low-income segment")

            if 'age' in stats:
                age = stats['age']['mean']
                if age >= 55:
                    cluster_insights.append("Senior customers - Stable income")
                elif age >= 35:
                    cluster_insights.append("Mid-career professionals")
                else:
                    cluster_insights.append("Young professionals - Growth potential")

            insights[cluster_name] = cluster_insights

        return insights

# Initialize the segmentation engine
@st.cache_resource
def get_segmentation_engine():
    return CustomerSegmentationEngine()

# Main page content
st.title("👥 Customer Segmentation")
st.markdown("### Advanced ML-powered Customer Clustering and Analysis")

# Initialize session state
if 'segmentation_data' not in st.session_state:
    st.session_state.segmentation_data = None
if 'segmentation_engine' not in st.session_state:
    st.session_state.segmentation_engine = get_segmentation_engine()

# Sidebar controls
st.sidebar.header("🔧 Segmentation Controls")

# Data upload section
st.sidebar.subheader("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload customer data",
    type=['csv', 'xlsx'],
    help="Upload CSV or Excel file with customer data"
)

# Generate sample data if no file uploaded
if uploaded_file is None:
    if st.sidebar.button("🎲 Generate Sample Data"):
        np.random.seed(42)
        n_customers = 1000

        sample_data = pd.DataFrame({
            'customer_id': range(1, n_customers + 1),
            'age': np.random.normal(40, 12, n_customers).astype(int),
            'income': np.random.lognormal(10.5, 0.5, n_customers).astype(int),
            'credit_score': np.random.normal(650, 100, n_customers).astype(int),
            'debt_to_income_ratio': np.random.uniform(0.1, 0.8, n_customers),
            'employment_length': np.random.exponential(5, n_customers),
            'loan_amount': np.random.lognormal(9, 0.8, n_customers).astype(int),
            'interest_rate': np.random.uniform(3, 18, n_customers),
            'num_credit_lines': np.random.poisson(3, n_customers),
            'credit_utilization': np.random.uniform(0.1, 0.9, n_customers)
        })

        # Ensure realistic ranges
        sample_data['age'] = np.clip(sample_data['age'], 18, 80)
        sample_data['credit_score'] = np.clip(sample_data['credit_score'], 300, 850)
        sample_data['employment_length'] = np.clip(sample_data['employment_length'], 0, 40)

        st.session_state.segmentation_data = sample_data
        st.session_state.segmentation_engine.load_data(sample_data)
        st.success("✅ Sample data generated successfully!")

# Load uploaded data
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        st.session_state.segmentation_data = data
        st.session_state.segmentation_engine.load_data(data)
        st.success(f"✅ Data loaded successfully! ({len(data)} rows, {len(data.columns)} columns)")

    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

# Main content area
if st.session_state.segmentation_data is not None:
    data = st.session_state.segmentation_data
    engine = st.session_state.segmentation_engine

    # Data overview
    st.subheader("📊 Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(data))
    with col2:
        st.metric("Features", len(data.columns))
    with col3:
        st.metric("Missing Values", data.isnull().sum().sum())
    with col4:
        st.metric("Data Quality", f"{((1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100):.1f}%")

    # Feature selection
    st.subheader("🎯 Feature Selection")

    # Get numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'customer_id' in numerical_cols:
        numerical_cols.remove('customer_id')

    selected_features = st.multiselect(
        "Select features for segmentation:",
        numerical_cols,
        default=numerical_cols[:5] if len(numerical_cols) >= 5 else numerical_cols,
        help="Choose numerical features that best represent customer characteristics"
    )

    if len(selected_features) >= 2:
        # Preprocessing options
        st.subheader("⚙️ Preprocessing Options")

        col1, col2 = st.columns(2)
        with col1:
            scaling_method = st.selectbox(
                "Scaling Method:",
                ['standard', 'minmax'],
                help="Standard: mean=0, std=1 | MinMax: range [0,1]"
            )

        with col2:
            handle_missing = st.selectbox(
                "Handle Missing Values:",
                ['median', 'mean', 'drop'],
                help="Method to handle missing values"
            )

        # Preprocess data
        success, message = engine.preprocess_data(selected_features, scaling_method)

        if success:
            st.success(f"✅ {message}")

            # Clustering algorithm selection
            st.subheader("🤖 Clustering Algorithm")

            col1, col2 = st.columns(2)
            with col1:
                algorithm = st.selectbox(
                    "Select Algorithm:",
                    ['kmeans', 'gaussian_mixture', 'hierarchical', 'dbscan'],
                    help="Choose clustering algorithm"
                )

            with col2:
                if algorithm != 'dbscan':
                    n_clusters = st.slider(
                        "Number of Clusters:",
                        min_value=2,
                        max_value=10,
                        value=5,
                        help="Number of customer segments to create"
                    )
                else:
                    st.write("DBSCAN automatically determines clusters")
                    n_clusters = None

            # DBSCAN specific parameters
            if algorithm == 'dbscan':
                col1, col2 = st.columns(2)
                with col1:
                    eps = st.slider("Epsilon (eps):", 0.1, 2.0, 0.5, 0.1)
                with col2:
                    min_samples = st.slider("Min Samples:", 2, 20, 5)

            # Optimal clusters analysis
            if algorithm == 'kmeans':
                st.subheader("📈 Optimal Clusters Analysis")

                if st.button("🔍 Find Optimal Clusters"):
                    with st.spinner("Analyzing optimal number of clusters..."):
                        cluster_range, metrics = engine.find_optimal_clusters(max_clusters=10)

                        # Create subplots for metrics
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Elbow Method (Inertia)', 'Silhouette Score', 
                                          'Calinski-Harabasz Score', 'Davies-Bouldin Score'),
                            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                   [{"secondary_y": False}, {"secondary_y": False}]]
                        )

                        # Elbow method
                        fig.add_trace(
                            go.Scatter(x=list(cluster_range), y=metrics['inertia'], 
                                     mode='lines+markers', name='Inertia'),
                            row=1, col=1
                        )

                        # Silhouette score
                        fig.add_trace(
                            go.Scatter(x=list(cluster_range), y=metrics['silhouette'], 
                                     mode='lines+markers', name='Silhouette'),
                            row=1, col=2
                        )

                        # Calinski-Harabasz score
                        fig.add_trace(
                            go.Scatter(x=list(cluster_range), y=metrics['calinski_harabasz'], 
                                     mode='lines+markers', name='Calinski-Harabasz'),
                            row=2, col=1
                        )

                        # Davies-Bouldin score
                        fig.add_trace(
                            go.Scatter(x=list(cluster_range), y=metrics['davies_bouldin'], 
                                     mode='lines+markers', name='Davies-Bouldin'),
                            row=2, col=2
                        )

                        fig.update_layout(height=600, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                        # Recommendations
                        best_silhouette = cluster_range[np.argmax(metrics['silhouette'])]
                        best_calinski = cluster_range[np.argmax(metrics['calinski_harabasz'])]
                        best_davies = cluster_range[np.argmin(metrics['davies_bouldin'])]

                        st.info(f"""
                        **Optimal Clusters Recommendations:**
                        - Silhouette Score: {best_silhouette} clusters
                        - Calinski-Harabasz: {best_calinski} clusters  
                        - Davies-Bouldin: {best_davies} clusters
                        """)

            # Perform clustering
            st.subheader("🎯 Perform Clustering")

            if st.button("🚀 Run Clustering Analysis", type="primary"):
                with st.spinner("Performing clustering analysis..."):
                    # Run clustering
                    if algorithm == 'dbscan':
                        success, message = engine.perform_clustering(
                            algorithm=algorithm, 
                            eps=eps, 
                            min_samples=min_samples
                        )
                    else:
                        success, message = engine.perform_clustering(
                            algorithm=algorithm, 
                            n_clusters=n_clusters
                        )

                    if success:
                        st.success(f"✅ {message}")

                        # Display clustering results
                        st.subheader("📊 Clustering Results")

                        # Cluster distribution
                        cluster_counts = pd.Series(engine.cluster_labels).value_counts().sort_index()

                        col1, col2 = st.columns(2)

                        with col1:
                            # Cluster size pie chart
                            fig_pie = px.pie(
                                values=cluster_counts.values,
                                names=[f'Cluster {i}' for i in cluster_counts.index],
                                title="Cluster Distribution"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                        with col2:
                            # Cluster size bar chart
                            fig_bar = px.bar(
                                x=[f'Cluster {i}' for i in cluster_counts.index],
                                y=cluster_counts.values,
                                title="Cluster Sizes"
                            )
                            st.plotly_chart(fig_bar, use_container_width=True)

                        # Cluster visualization using PCA
                        st.subheader("🎨 Cluster Visualization")

                        # PCA for 2D visualization
                        pca = PCA(n_components=2, random_state=42)
                        pca_data = pca.fit_transform(engine.scaled_data)

                        # Create visualization dataframe
                        viz_df = pd.DataFrame({
                            'PC1': pca_data[:, 0],
                            'PC2': pca_data[:, 1],
                            'Cluster': [f'Cluster {i}' for i in engine.cluster_labels]
                        })

                        fig_scatter = px.scatter(
                            viz_df, x='PC1', y='PC2', color='Cluster',
                            title=f'Customer Segments Visualization (PCA)',
                            labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                                   'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'}
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)

                        # Cluster profiles
                        st.subheader("📋 Cluster Profiles")

                        profiles = engine.get_cluster_profiles()
                        insights = engine.get_cluster_insights()

                        if profiles:
                            for cluster_name, profile in profiles.items():
                                with st.expander(f"📊 {cluster_name} ({profile['size']} customers, {profile['percentage']:.1f}%)"):

                                    # Business insights
                                    if insights and cluster_name in insights:
                                        st.markdown("**🎯 Business Insights:**")
                                        for insight in insights[cluster_name]:
                                            st.markdown(f"• {insight}")
                                        st.markdown("---")

                                    # Statistical profile
                                    st.markdown("**📊 Statistical Profile:**")

                                    profile_df = pd.DataFrame(profile['statistics']).T
                                    profile_df = profile_df.round(2)
                                    st.dataframe(profile_df, use_container_width=True)

                                    # Feature comparison chart
                                    means = [profile['statistics'][feat]['mean'] for feat in selected_features]

                                    fig_radar = go.Figure()
                                    fig_radar.add_trace(go.Scatterpolar(
                                        r=means,
                                        theta=selected_features,
                                        fill='toself',
                                        name=cluster_name
                                    ))

                                    fig_radar.update_layout(
                                        polar=dict(
                                            radialaxis=dict(visible=True)
                                        ),
                                        showlegend=True,
                                        title=f"{cluster_name} Feature Profile"
                                    )

                                    st.plotly_chart(fig_radar, use_container_width=True)

                        # Feature importance heatmap
                        st.subheader("🔥 Feature Importance Heatmap")

                        # Create feature importance matrix
                        cluster_means = []
                        cluster_names = []

                        for cluster_id in sorted(np.unique(engine.cluster_labels)):
                            if cluster_id == -1:  # Skip noise points in DBSCAN
                                continue
                            cluster_data = data[data['Cluster'] == cluster_id]
                            means = [cluster_data[feat].mean() for feat in selected_features]
                            cluster_means.append(means)
                            cluster_names.append(f'Cluster {cluster_id}')

                        if cluster_means:
                            heatmap_data = np.array(cluster_means)

                            fig_heatmap = px.imshow(
                                heatmap_data,
                                x=selected_features,
                                y=cluster_names,
                                aspect='auto',
                                title='Cluster Feature Heatmap (Mean Values)'
                            )
                            st.plotly_chart(fig_heatmap, use_container_width=True)

                        # Export results
                        st.subheader("📤 Export Results")

                        col1, col2 = st.columns(2)

                        with col1:
                            # Download clustered data
                            csv_data = data.to_csv(index=False)
                            st.download_button(
                                label="📊 Download Clustered Data (CSV)",
                                data=csv_data,
                                file_name="customer_segments.csv",
                                mime="text/csv"
                            )

                        with col2:
                            # Download cluster profiles
                            if profiles:
                                profiles_json = pd.DataFrame(profiles).to_json(indent=2)
                                st.download_button(
                                    label="📋 Download Cluster Profiles (JSON)",
                                    data=profiles_json,
                                    file_name="cluster_profiles.json",
                                    mime="application/json"
                                )

                    else:
                        st.error(f"❌ {message}")

        else:
            st.error(f"❌ {message}")

    else:
        st.warning("⚠️ Please select at least 2 features for segmentation")

else:
    st.info("📁 Please upload customer data or generate sample data to begin segmentation analysis")

    # Show example data format
    st.subheader("📋 Expected Data Format")

    example_data = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'age': [25, 35, 45, 55, 30],
        'income': [50000, 75000, 100000, 120000, 60000],
        'credit_score': [650, 720, 780, 800, 680],
        'debt_to_income_ratio': [0.3, 0.2, 0.15, 0.1, 0.25],
        'employment_length': [2, 8, 15, 20, 5]
    })

    st.dataframe(example_data, use_container_width=True)

    st.markdown("""
    **Required columns:**
    - Numerical features representing customer characteristics
    - At least 2 features for meaningful segmentation
    - Clean data with minimal missing values

    **Recommended features:**
    - Demographics: age, income, employment_length
    - Credit profile: credit_score, debt_to_income_ratio
    - Behavioral: spending_patterns, payment_history
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Customer Segmentation Engine v2.0 | Advanced ML-powered clustering and analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)
