import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import traceback
import boto3
from botocore.exceptions import NoCredentialsError
import json
import io

# Import the improved calculator
class DemoRDSSizingCalculator:
    ENGINES = ['oracle-ee', 'oracle-se', 'postgres', 'aurora-postgresql', 'aurora-mysql', 'sqlserver']
    
    ENV_PROFILES = {
        "PROD": {"cpu_factor": 1.0, "ram_factor": 1.0, "storage_factor": 1.0, "performance_headroom": 1.2},
        "SQA": {"cpu_factor": 0.7, "ram_factor": 0.75, "storage_factor": 0.7, "performance_headroom": 1.1},
        "QA": {"cpu_factor": 0.5, "ram_factor": 0.6, "storage_factor": 0.5, "performance_headroom": 1.05},
        "DEV": {"cpu_factor": 0.25, "ram_factor": 0.4, "storage_factor": 0.3, "performance_headroom": 1.0}
    }
    
    def __init__(self):
        self.aws_available = self._check_aws_credentials()
        self.recommendations = {}
        self.inputs = {}
        self.bulk_recommendations = {}
    
    def _check_aws_credentials(self):
        """Check if AWS credentials are available"""
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            return credentials is not None
        except Exception:
            return False
            
    def refresh_aws_credentials(self):
        """Refresh AWS credentials and return status"""
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            self.aws_available = credentials is not None
            return self.aws_available
        except Exception as e:
            self.aws_available = False
            return False
            
    def _get_instance_data(self, engine, region):
        """Get instance data - real-time if available, fallback otherwise"""
        instances = {
            "postgres": [
                {"type": "db.t3.micro", "vCPU": 2, "memory": 1, "pricing": {"ondemand": 0.026}},
                {"type": "db.t3.small", "vCPU": 2, "memory": 2, "pricing": {"ondemand": 0.051}},
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.102}},
                {"type": "db.t3.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.204}},
                {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.192}},
                {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand": 0.384}},
                {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "pricing": {"ondemand": 0.768}},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.24}},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 0.48}},
            ],
            "oracle-ee": [
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.272}},
                {"type": "db.t3.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.544}},
                {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.475}},
                {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand": 0.95}},
                {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "pricing": {"ondemand": 1.90}},
                {"type": "db.m5.4xlarge", "vCPU": 16, "memory": 64, "pricing": {"ondemand": 3.80}},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.60}},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 1.20}},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand": 2.40}},
            ],
            "aurora-postgresql": [
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.082}},
                {"type": "db.t4g.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.073}},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.285}},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 0.57}},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand": 1.14}},
                {"type": "db.r6g.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.256}},
                {"type": "db.r6g.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 0.512}},
                {"type": "db.serverless", "vCPU": 0, "memory": 0, "pricing": {"ondemand": 0.12}},
            ]
        }
        
        return instances.get(engine, instances["postgres"])
    
    def calculate_requirements(self, env):
        """Calculate requirements for a specific environment"""
        profile = self.ENV_PROFILES[env]
        
        # Calculate base requirements
        base_vcpus = self.inputs["on_prem_cores"] * (self.inputs["peak_cpu_percent"] / 100)
        base_ram = self.inputs["on_prem_ram_gb"] * (self.inputs["peak_ram_percent"] / 100)
        
        # Apply environment-specific factors
        required_vcpus = max(1, int(base_vcpus * profile["cpu_factor"] * profile["performance_headroom"]))
        required_ram = max(1, int(base_ram * profile["ram_factor"] * profile["performance_headroom"]))
        
        # Environment minimums
        env_minimums = {
            "PROD": {"cpu": 4, "ram": 8},
            "SQA": {"cpu": 2, "ram": 4}, 
            "QA": {"cpu": 2, "ram": 4},
            "DEV": {"cpu": 1, "ram": 2}
        }
        
        min_reqs = env_minimums[env]
        required_vcpus = max(required_vcpus, min_reqs["cpu"])
        required_ram = max(required_ram, min_reqs["ram"])
        
        # Get available instances
        instances = self._get_instance_data(self.inputs["engine"], self.inputs["region"])
        
        # Select optimal instance
        selected_instance = self._select_instance(required_vcpus, required_ram, env, instances)
        
        # Calculate storage
        storage = self._calculate_storage(env)
        
        # Calculate costs
        costs = self._calculate_costs(selected_instance, storage, env)
        
        # Generate advisories
        advisories = self._generate_advisories(selected_instance, required_vcpus, required_ram, env)
        
        return {
            "environment": env,
            "instance_type": selected_instance["type"],
            "vCPUs": required_vcpus,
            "RAM_GB": required_ram,
            "actual_vCPUs": selected_instance["vCPU"],
            "actual_RAM_GB": selected_instance["memory"],
            "storage_GB": storage,
            "total_cost": costs["total"],
            "instance_cost": costs["instance"],
            "storage_cost": costs["storage"],
            "advisories": advisories
        }
    
    def _select_instance(self, required_vcpus, required_ram, env, instances):
        """Select optimal instance based on requirements and environment"""
        suitable = []
        
        for instance in instances:
            if instance["vCPU"] >= required_vcpus and instance["memory"] >= required_ram:
                suitable.append(instance)
        
        if not suitable:
            # Relaxed matching for dev/test
            tolerance = 0.8 if env in ["DEV", "QA"] else 0.9
            for instance in instances:
                if (instance["vCPU"] >= required_vcpus * tolerance and 
                    instance["memory"] >= required_ram * tolerance):
                    suitable.append(instance)
        
        if not suitable:
            return instances[-1]  # Return largest if nothing fits
        
        # Selection strategy
        if env == "PROD":
            # Balance performance and cost for production
            def score(inst):
                headroom = (inst["vCPU"] + inst["memory"]) / (required_vcpus + required_ram)
                cost_factor = 1000 / (inst["pricing"]["ondemand"] + 1)
                return min(headroom, 2.0) * 0.7 + cost_factor * 0.3
            return max(suitable, key=score)
        else:
            # Cost-optimize for non-production
            return min(suitable, key=lambda x: x["pricing"]["ondemand"])
    
    def _calculate_storage(self, env):
        """Calculate storage requirements"""
        profile = self.ENV_PROFILES[env]
        base_storage = self.inputs["storage_current_gb"]
        return max(20, int(base_storage * profile["storage_factor"] * 1.3))
    
    def _calculate_costs(self, instance, storage, env):
        """Calculate monthly costs"""
        hourly = instance["pricing"]["ondemand"]
        
        # Apply deployment factor
        deployment_factors = {"Single-AZ": 1, "Multi-AZ": 2, "Multi-AZ Cluster": 2.5}
        deploy_factor = deployment_factors.get(self.inputs.get("deployment", "Multi-AZ"), 2)
        
        instance_cost = hourly * 24 * 30 * deploy_factor
        storage_cost = storage * 0.10  # $0.10/GB/month
        backup_cost = storage * 0.095 * 0.25  # Approximate backup cost
        
        total_cost = instance_cost + storage_cost + backup_cost
        
        return {
            "instance": instance_cost,
            "storage": storage_cost,
            "backup": backup_cost,
            "total": total_cost
        }
    
    def _generate_advisories(self, instance, required_vcpus, required_ram, env):
        """Generate optimization advisories"""
        advisories = []
        
        cpu_ratio = instance["vCPU"] / max(required_vcpus, 1)
        ram_ratio = instance["memory"] / max(required_ram, 1)
        
        if cpu_ratio > 2:
            advisories.append(f"⚠️ CPU over-provisioned: {instance['vCPU']} vs {required_vcpus} needed")
        
        if ram_ratio > 2:
            advisories.append(f"⚠️ RAM over-provisioned: {instance['memory']}GB vs {required_ram}GB needed")
        
        if env == "PROD" and self.inputs.get("deployment") == "Single-AZ":
            advisories.append("🚨 Use Multi-AZ for production high availability")
        
        if env in ["DEV", "QA"] and instance["pricing"]["ondemand"] > 1.0:
            advisories.append("💡 Consider smaller instances for dev/test environments")
        
        return advisories
    
    def generate_all_recommendations(self):
        """Generate recommendations for all environments"""
        self.recommendations = {}
        
        for env in self.ENV_PROFILES:
            try:
                self.recommendations[env] = self.calculate_requirements(env)
            except Exception as e:
                self.recommendations[env] = {"error": str(e)}
        
        return self.recommendations
    
    def process_bulk_workloads(self, workload_data, global_settings):
        """Process multiple workloads from uploaded data"""
        self.bulk_recommendations = {}
        
        for index, row in workload_data.iterrows():
            workload_name = row.get('workload_name', f"Workload_{index + 1}")
            
            # Set inputs for this workload
            self.inputs = {
                "region": global_settings["region"],
                "engine": global_settings["engine"],
                "deployment": global_settings["deployment"],
                "on_prem_cores": int(row["cpu_cores"]),
                "peak_cpu_percent": float(row["peak_cpu_percent"]),
                "on_prem_ram_gb": int(row["ram_gb"]),
                "peak_ram_percent": float(row["peak_ram_percent"]),
                "storage_current_gb": int(row["storage_gb"]),
                "storage_growth_rate": float(row.get("growth_rate_percent", 20)) / 100
            }
            
            try:
                workload_recommendations = self.generate_all_recommendations()
                self.bulk_recommendations[workload_name] = workload_recommendations
            except Exception as e:
                self.bulk_recommendations[workload_name] = {"error": str(e)}
        
        return self.bulk_recommendations

# Configure Streamlit
st.set_page_config(
    page_title="Enhanced AWS RDS Sizing Tool with Bulk Upload",
    layout="wide",
    page_icon="🚀"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #111;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        color: #eee;
    }
    .advisory-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .status-good {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #155724;
    }
    .status-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        margin: 0.5rem 0;
        color: #856404;
    }
    .status-error {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .metric-cost {
        color: #111 !important;
        font-size: 2.2rem;
    }
    .upload-section {
        background: #f8f9fa;
        border: 2px dashed #6c757d;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .sample-template {
        background: #e7f3ff;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize calculator
@st.cache_resource
def get_calculator():
    return DemoRDSSizingCalculator()

calculator = get_calculator()

# Header
st.title("🚀 Enhanced AWS RDS & Aurora Sizing Tool")
st.markdown("**Real-time AWS pricing integration with bulk upload support**")

# Mode Selection
mode = st.radio(
    "Select Mode:",
    ["Single Workload", "Bulk Upload"],
    horizontal=True,
    help="Choose between single workload analysis or bulk processing of multiple workloads"
)

# AWS Status Check
col1, col2 = st.columns([3, 1])
with col1:
    if calculator.aws_available:
        st.markdown('<div class="status-good">✅ AWS credentials detected - real-time pricing available</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">⚠️ AWS credentials not found - using fallback pricing data</div>', unsafe_allow_html=True)

with col2:
    if st.button("🔄 Refresh AWS Status"):
        with st.spinner("Checking AWS credentials..."):
            success = calculator.refresh_aws_credentials()
            
        if success:
            st.success("✅ AWS credentials verified successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to find valid AWS credentials. Using fallback data.")
            time.sleep(2)
            st.rerun()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # AWS Settings
    with st.expander("☁️ AWS Settings", expanded=True):
        region = st.selectbox("Region", ["us-east-1", "us-west-1", "us-west-2", "eu-west-1"], index=0)
        engine = st.selectbox("Database Engine", calculator.ENGINES, index=2)
        deployment = st.selectbox("Deployment", ["Single-AZ", "Multi-AZ", "Multi-AZ Cluster"], index=1)
    
    if mode == "Single Workload":
        # Workload Profile
        with st.expander("📊 Current Workload", expanded=True):
            cores = st.number_input("CPU Cores", min_value=1, max_value=128, value=8, step=1)
            cpu_util = st.slider("Peak CPU %", 1, 100, 70)
            ram = st.number_input("RAM (GB)", min_value=1, max_value=1024, value=32, step=1)
            ram_util = st.slider("Peak RAM %", 1, 100, 80)
        
        with st.expander("💾 Storage", expanded=True):
            storage = st.number_input("Current Storage (GB)", min_value=1, value=250)
            growth = st.number_input("Annual Growth %", min_value=0, max_value=100, value=20)
    
    # Real-time pricing toggle
    st.header("⚡ Performance")
    use_realtime = st.checkbox("Use Real-time AWS Pricing", value=calculator.aws_available)
    if use_realtime and not calculator.aws_available:
        st.warning("⚠️ AWS credentials required for real-time pricing")

# Main Content based on mode
if mode == "Single Workload":
    # Update calculator inputs
    calculator.inputs = {
        "region": region,
        "engine": engine,
        "deployment": deployment,
        "on_prem_cores": cores,
        "peak_cpu_percent": cpu_util,
        "on_prem_ram_gb": ram,
        "peak_ram_percent": ram_util,
        "storage_current_gb": storage,
        "storage_growth_rate": growth/100
    }

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🚀 Generate Sizing Recommendations", type="primary", use_container_width=True):
            with st.spinner("🔄 Generating recommendations..."):
                start_time = time.time()
                
                try:
                    results = calculator.generate_all_recommendations()
                    st.session_state['results'] = results
                    st.session_state['generation_time'] = time.time() - start_time
                    st.session_state['mode'] = 'single'
                    
                    # Verify recommendation diversity
                    instance_types = [r.get('instance_type', 'N/A') for r in results.values() if 'error' not in r]
                    unique_types = len(set(instance_types))
                    
                    if unique_types == 1 and len(instance_types) > 1:
                        st.warning(f"⚠️ All environments received same instance type: {instance_types[0]}")
                    else:
                        st.success(f"✅ Generated diverse recommendations: {unique_types} different instance types")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

    with col2:
        export_btn = st.button("📊 Export Results", use_container_width=True)

    with col3:
        if st.button("🔧 Debug Info", use_container_width=True):
            st.session_state['show_debug'] = not st.session_state.get('show_debug', False)

else:  # Bulk Upload Mode
    st.header("📁 Bulk Workload Upload")
    
    # Sample Template Information
    st.markdown("""
    <div class="sample-template">
        <h4>📋 CSV Template Format</h4>
        <p>Your CSV file should contain the following columns:</p>
        <ul>
            <li><strong>workload_name</strong> (optional): Name for the workload</li>
            <li><strong>cpu_cores</strong>: Number of CPU cores</li>
            <li><strong>peak_cpu_percent</strong>: Peak CPU utilization percentage</li>
            <li><strong>ram_gb</strong>: RAM in GB</li>
            <li><strong>peak_ram_percent</strong>: Peak RAM utilization percentage</li>
            <li><strong>storage_gb</strong>: Current storage in GB</li>
            <li><strong>growth_rate_percent</strong> (optional): Annual growth rate percentage (default: 20)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Download template
    sample_data = pd.DataFrame({
        'workload_name': ['Web App DB', 'Analytics DB', 'Dev Environment'],
        'cpu_cores': [8, 16, 4],
        'peak_cpu_percent': [70, 85, 50],
        'ram_gb': [32, 64, 16],
        'peak_ram_percent': [80, 90, 60],
        'storage_gb': [250, 500, 100],
        'growth_rate_percent': [20, 30, 10]
    })
    
    csv_template = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV Template",
        data=csv_template,
        file_name="rds_sizing_template.csv",
        mime="text/csv"
    )
    
    # File Upload
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file containing multiple workloads to analyze"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                workload_data = pd.read_csv(uploaded_file)
            else:
                workload_data = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(workload_data)} workloads.")
            
            # Display preview
            st.subheader("📊 Data Preview")
            st.dataframe(workload_data.head(), use_container_width=True)
            
            # Validate required columns
            required_columns = ['cpu_cores', 'peak_cpu_percent', 'ram_gb', 'peak_ram_percent', 'storage_gb']
            missing_columns = [col for col in required_columns if col not in workload_data.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
            else:
                # Process bulk workloads
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if st.button("🚀 Process All Workloads", type="primary", use_container_width=True):
                        with st.spinner(f"🔄 Processing {len(workload_data)} workloads..."):
                            start_time = time.time()
                            
                            global_settings = {
                                "region": region,
                                "engine": engine,
                                "deployment": deployment
                            }
                            
                            try:
                                bulk_results = calculator.process_bulk_workloads(workload_data, global_settings)
                                st.session_state['bulk_results'] = bulk_results
                                st.session_state['bulk_generation_time'] = time.time() - start_time
                                st.session_state['mode'] = 'bulk'
                                st.session_state['workload_count'] = len(workload_data)
                                
                                st.success(f"✅ Processed {len(bulk_results)} workloads successfully!")
                                
                            except Exception as e:
                                st.error(f"❌ Error processing workloads: {str(e)}")
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc())
                
                with col2:
                    bulk_export_btn = st.button("📊 Export Bulk Results", use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

# Display Results based on mode
if st.session_state.get('mode') == 'single' and 'results' in st.session_state:
    results = st.session_state['results']
    
    # Summary Metrics
    st.header("📊 Recommendation Summary")
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        prod_result = valid_results.get('PROD', list(valid_results.values())[0])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{prod_result['instance_type']}</div>
                <div class="metric-label">Production Instance</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{prod_result['actual_vCPUs']} / {prod_result['actual_RAM_GB']}GB</div>
                <div class="metric-label">vCPUs / RAM</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value metric-cost">${prod_result['total_cost']:,.0f}</div>
                <div class="metric-label">Monthly Cost</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            generation_time = st.session_state.get('generation_time', 0)
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{generation_time:.1f}s</div>
                <div class="metric-label">Generation Time</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed Results Table
        st.header("📋 Environment-Specific Recommendations")
        
        df_data = []
        for env, result in valid_results.items():
            df_data.append({
                'Environment': env,
                'Instance Type': result['instance_type'],
                'Required vCPUs': result['vCPUs'],
                'Actual vCPUs': result['actual_vCPUs'],
                'Required RAM (GB)': result['RAM_GB'],
                'Actual RAM (GB)': result['actual_RAM_GB'],
                'Storage (GB)': result['storage_GB'],
                'Monthly Cost': result['total_cost']
            })
        
        df = pd.DataFrame(df_data)
        
        # Color-code the dataframe
        def highlight_costs(val):
            if val < 500:
                return 'background-color: #d4edda'
            elif val < 1500:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'
        
        styled_df = df.style.applymap(highlight_costs, subset=['Monthly Cost'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Cost Comparison Chart
        st.header("💰 Cost Comparison by Environment")
        
        fig = px.bar(
            df, 
            x='Environment', 
            y='Monthly Cost',
            color='Environment',
            title='Monthly Cost by Environment',
            text='Monthly Cost'
        )
        fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title="Monthly Cost ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Resource Utilization Chart
        st.header("⚡ Resource Allocation vs Requirements")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('CPU Allocation', 'RAM Allocation'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # CPU Chart
        fig.add_trace(
            go.Bar(name='Required', x=df['Environment'], y=df['Required vCPUs'], marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Allocated', x=df['Environment'], y=df['Actual vCPUs'], marker_color='darkblue'),
            row=1, col=1
        )
        
        # RAM Chart  
        fig.add_trace(
            go.Bar(name='Required', x=df['Environment'], y=df['Required RAM (GB)'], marker_color='lightcoral', showlegend=False),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Allocated', x=df['Environment'], y=df['Actual RAM (GB)'], marker_color='darkred', showlegend=False),
            row=1, col=2
        )
        
        fig.update_layout(height=400, barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimization Advisories
        st.header("💡 Optimization Advisories")
        
        advisories_found = False
        for env, result in valid_results.items():
            if result.get('advisories'):
                advisories_found = True
                st.subheader(f"{env} Environment")
                for advisory in result['advisories']:
                    st.markdown(f'<div class="advisory-box">{advisory}</div>', unsafe_allow_html=True)
        
        if not advisories_found:
            st.success("✅ No optimization advisories - your configuration looks optimal!")
        
        # Error Summary
        error_results = {k: v for k, v in results.items() if 'error' in v}
        if error_results:
            st.header("❌ Errors")
            for env, result in error_results.items():
                st.error(f"{env}: {result['error']}")

elif st.session_state.get('mode') == 'bulk' and 'bulk_results' in st.session_state:
    bulk_results = st.session_state['bulk_results']
    
    # Bulk Summary Metrics
    st.header("📊 Bulk Processing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    workload_count = st.session_state.get('workload_count', 0)
    generation_time = st.session_state.get('bulk_generation_time', 0)
    
    # Calculate total costs across all workloads for PROD environment
    total_prod_cost = 0
    successful_workloads = 0
    
    for workload_name, workload_recommendations in bulk_results.items():
        if 'error' not in workload_recommendations and 'PROD' in workload_recommendations:
            total_prod_cost += workload_recommendations['PROD'].get('total_cost', 0)
            successful_workloads += 1
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{workload_count}</div>
            <div class="metric-label">Total Workloads</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{successful_workloads}</div>
            <div class="metric-label">Successful</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value metric-cost">${total_prod_cost:,.0f}</div>
            <div class="metric-label">Total PROD Cost/Month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{generation_time:.1f}s</div>
            <div class="metric-label">Processing Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Bulk Results Table
    st.header("📋 Bulk Recommendations Summary")
    
    bulk_df_data = []
    for workload_name, workload_recommendations in bulk_results.items():
        if 'error' not in workload_recommendations:
            for env in ['PROD', 'SQA', 'QA', 'DEV']:
                if env in workload_recommendations:
                    result = workload_recommendations[env]
                    bulk_df_data.append({
                        'Workload': workload_name,
                        'Environment': env,
                        'Instance Type': result['instance_type'],
                        'vCPUs': result['actual_vCPUs'],
                        'RAM (GB)': result['actual_RAM_GB'],
                        'Storage (GB)': result['storage_GB'],
                        'Monthly Cost': result['total_cost']
                    })
    
    if bulk_df_data:
        bulk_df = pd.DataFrame(bulk_df_data)
        
        # Add filtering options
        col1, col2 = st.columns(2)
        with col1:
            selected_workloads = st.multiselect(
                "Filter Workloads:",
                options=bulk_df['Workload'].unique(),
                default=bulk_df['Workload'].unique()
            )
        
        with col2:
            selected_envs = st.multiselect(
                "Filter Environments:",
                options=['PROD', 'SQA', 'QA', 'DEV'],
                default=['PROD', 'SQA', 'QA', 'DEV']
            )
        
        # Filter dataframe
        filtered_df = bulk_df[
            (bulk_df['Workload'].isin(selected_workloads)) & 
            (bulk_df['Environment'].isin(selected_envs))
        ]
        
        # Display filtered results
        st.dataframe(filtered_df, use_container_width=True)
        
        # Bulk Visualization
        st.header("📊 Bulk Analysis Charts")
        
        # Cost by Workload and Environment
        fig = px.bar(
            filtered_df,
            x='Workload',
            y='Monthly Cost',
            color='Environment',
            title='Monthly Cost by Workload and Environment',
            barmode='group'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Instance Type Distribution
        instance_dist = filtered_df.groupby(['Environment', 'Instance Type']).size().reset_index(name='Count')
        fig2 = px.sunburst(
            instance_dist,
            path=['Environment', 'Instance Type'],
            values='Count',
            title='Instance Type Distribution by Environment'
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # Cost Summary by Environment
        cost_summary = filtered_df.groupby('Environment')['Monthly Cost'].agg(['sum', 'mean', 'count']).reset_index()
        cost_summary.columns = ['Environment', 'Total Cost', 'Average Cost', 'Workload Count']
        
        st.subheader("💰 Cost Summary by Environment")
        st.dataframe(cost_summary, use_container_width=True)
    
    # Error Summary for bulk processing
    error_workloads = {k: v for k, v in bulk_results.items() if 'error' in v}
    if error_workloads:
        st.header("❌ Processing Errors")
        for workload, error_info in error_workloads.items():
            st.error(f"{workload}: {error_info['error']}")

# Export functionality
if st.session_state.get('mode') == 'single' and export_btn and 'results' in st.session_state:
    results = st.session_state['results']
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        df_export = pd.DataFrame([
            {
                'Environment': env,
                'Instance Type': result['instance_type'],
                'vCPUs': result['actual_vCPUs'],
                'RAM (GB)': result['actual_RAM_GB'],
                'Storage (GB)': result['storage_GB'],
                'Monthly Cost': result['total_cost']
            }
            for env, result in valid_results.items()
        ])
        
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"rds_sizing_{engine}_{region}_{int(time.time())}.csv",
            mime="text/csv"
        )

elif st.session_state.get('mode') == 'bulk' and 'bulk_export_btn' in locals() and bulk_export_btn and 'bulk_results' in st.session_state:
    bulk_results = st.session_state['bulk_results']
    
    export_data = []
    for workload_name, workload_recommendations in bulk_results.items():
        if 'error' not in workload_recommendations:
            for env, result in workload_recommendations.items():
                export_data.append({
                    'Workload': workload_name,
                    'Environment': env,
                    'Instance Type': result['instance_type'],
                    'Required vCPUs': result['vCPUs'],
                    'Actual vCPUs': result['actual_vCPUs'],
                    'Required RAM (GB)': result['RAM_GB'],
                    'Actual RAM (GB)': result['actual_RAM_GB'],
                    'Storage (GB)': result['storage_GB'],
                    'Instance Cost': result['instance_cost'],
                    'Storage Cost': result['storage_cost'],
                    'Total Monthly Cost': result['total_cost']
                })
    
    if export_data:
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Bulk Results CSV",
            data=csv,
            file_name=f"rds_bulk_sizing_{engine}_{region}_{int(time.time())}.csv",
            mime="text/csv"
        )

# Debug Information
if st.session_state.get('show_debug', False):
    st.header("🔧 Debug Information")
    
    with st.expander("Calculator Inputs"):
        st.json(calculator.inputs)
    
    with st.expander("AWS Status"):
        st.write(f"AWS Available: {calculator.aws_available}")
        st.write(f"Region: {calculator.inputs.get('region', 'N/A')}")
        st.write(f"Engine: {calculator.inputs.get('engine', 'N/A')}")
    
    if st.session_state.get('mode') == 'single' and 'results' in st.session_state:
        with st.expander("Single Workload Results"):
            st.json(st.session_state['results'])
    
    if st.session_state.get('mode') == 'bulk' and 'bulk_results' in st.session_state:
        with st.expander("Bulk Results"):
            st.json(st.session_state['bulk_results'])

# Footer
st.markdown("---")
st.markdown("""
**🎯 Key Features:**
- ✅ Environment-specific sizing (PROD, SQA, QA, DEV with different resource factors)
- ✅ Single workload analysis with detailed recommendations
- ✅ Bulk upload and processing of multiple workloads
- ✅ CSV/Excel template download for easy data entry
- ✅ Comprehensive cost analysis and visualization
- ✅ Export capabilities for both single and bulk results
""")