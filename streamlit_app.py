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
    
    def _check_aws_credentials(self):
        """Check if AWS credentials are available"""
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            return credentials is not None
        except Exception:
            return False
            
    # CORRECTED INDENTATION FOR THIS METHOD
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
            
    # ... rest of the class remains the same ...
    
    def _get_instance_data(self, engine, region):
        """Get instance data - real-time if available, fallback otherwise"""
        # This would contain the real AWS API calls
        # For demo, using enhanced fallback data
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

# Configure Streamlit
st.set_page_config(
    page_title="Enhanced AWS RDS Sizing Tool",
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
        color: #111;  /* Changed to almost black */
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        color: #eee;  /* Light gray for labels */
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
    /* Add specific styling for cost values */
    .metric-cost {
        color: #111 !important;  /* Force black for cost values */
        font-size: 2.2rem;       /* Slightly larger for emphasis */
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
st.markdown("**Real-time AWS pricing integration with improved environment-specific recommendations**")

# AWS Status Check
col1, col2 = st.columns([3, 1])
with col1:
    if calculator.aws_available:
        st.markdown('<div class="status-good">✅ AWS credentials detected - real-time pricing available</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-warning">⚠️ AWS credentials not found - using fallback pricing data</div>', unsafe_allow_html=True)

with col2:
    if st.button("🔄 Refresh AWS Status"):
        # Show spinner while checking credentials
        with st.spinner("Checking AWS credentials..."):
            success = calculator.refresh_aws_credentials()
            
        if success:
            st.success("✅ AWS credentials verified successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to find valid AWS credentials. Using fallback data.")
            # Wait a moment so user can see the message
            time.sleep(2)
            st.rerun()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # AWS Settings
    with st.expander("☁️ AWS Settings", expanded=True):
        region = st.selectbox("Region", ["us-east-1", "us-west-1", "us-west-2", "eu-west-1"], index=0)
        engine = st.selectbox("Database Engine", calculator.ENGINES, index=2)  # Default to postgres
        deployment = st.selectbox("Deployment", ["Single-AZ", "Multi-AZ", "Multi-AZ Cluster"], index=1)
    
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

# Main Content
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if st.button("🚀 Generate Sizing Recommendations", type="primary", use_container_width=True):
        with st.spinner("🔄 Generating recommendations..."):
            start_time = time.time()
            
            try:
                results = calculator.generate_all_recommendations()
                st.session_state['results'] = results
                st.session_state['generation_time'] = time.time() - start_time
                
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

# Display Results
if 'results' in st.session_state:
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
                return 'background-color: #d4edda'  # Green for low cost
            elif val < 1500:
                return 'background-color: #fff3cd'  # Yellow for medium cost
            else:
                return 'background-color: #f8d7da'  # Red for high cost
        
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

# Debug Information
if st.session_state.get('show_debug', False):
    st.header("🔧 Debug Information")
    
    with st.expander("Calculator Inputs"):
        st.json(calculator.inputs)
    
    with st.expander("AWS Status"):
        st.write(f"AWS Available: {calculator.aws_available}")
        st.write(f"Region: {calculator.inputs.get('region', 'N/A')}")
        st.write(f"Engine: {calculator.inputs.get('engine', 'N/A')}")
    
    if 'results' in st.session_state:
        with st.expander("Raw Results"):
            st.json(st.session_state['results'])

# Export functionality
if export_btn and 'results' in st.session_state:
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

# Footer
st.markdown("---")
st.markdown("""
**🎯 Key Improvements:**
- ✅ Environment-specific sizing (PROD, SQA, QA, DEV with different resource factors)
- ✅ Refresh the app to enable real-time pricing
""")