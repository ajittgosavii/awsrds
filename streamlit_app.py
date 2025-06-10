import streamlit as st
import pandas as pd
import time
import traceback

# Try to import optional dependencies with fallbacks
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    st.warning("⚠️ Plotly not installed. Charts will be simplified. Install with: pip install plotly")
    HAS_PLOTLY = False

try:
    import boto3
    from botocore.exceptions import NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    st.warning("⚠️ Boto3 not installed. Real-time pricing unavailable. Install with: pip install boto3")
    HAS_BOTO3 = False

try:
    import json
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Simplified calculator class with dependency checks
class RobustRDSSizingCalculator:
    ENGINES = ['oracle-ee', 'oracle-se', 'postgres', 'aurora-postgresql', 'aurora-mysql', 'sqlserver']
    
    # Deployment options with reader/writer characteristics
    DEPLOYMENT_OPTIONS = {
        'Single-AZ': {
            'cost_multiplier': 1.0,
            'has_readers': False,
            'reader_count': 0,
            'description': 'Single instance deployment'
        },
        'Multi-AZ': {
            'cost_multiplier': 2.0,
            'has_readers': True,
            'reader_count': 1,
            'description': 'Primary + standby in different AZs'
        },
        'Multi-AZ Cluster': {
            'cost_multiplier': 2.5,
            'has_readers': True,
            'reader_count': 2,
            'description': 'Primary + 2 readers across AZs'
        },
        'Aurora Global': {
            'cost_multiplier': 3.0,
            'has_readers': True,
            'reader_count': 3,
            'description': 'Global database with cross-region readers'
        }
    }
    
    # Workload patterns
    WORKLOAD_PATTERNS = {
        'OLTP_BALANCED': {
            'read_percentage': 60,
            'write_percentage': 40,
            'description': 'Balanced OLTP workload'
        },
        'READ_HEAVY': {
            'read_percentage': 80,
            'write_percentage': 20,
            'description': 'Read-heavy analytical workload'
        },
        'WRITE_HEAVY': {
            'read_percentage': 30,
            'write_percentage': 70,
            'description': 'Write-heavy transactional workload'
        },
        'MIXED': {
            'read_percentage': 50,
            'write_percentage': 50,
            'description': 'Mixed read/write workload'
        }
    }
    
    ENV_PROFILES = {
        "PROD": {
            "cpu_factor": 1.0, "ram_factor": 1.0, "storage_factor": 1.0, 
            "performance_headroom": 1.2, "reader_sizing_factor": 1.0
        },
        "SQA": {
            "cpu_factor": 0.7, "ram_factor": 0.75, "storage_factor": 0.7, 
            "performance_headroom": 1.1, "reader_sizing_factor": 0.8
        },
        "QA": {
            "cpu_factor": 0.5, "ram_factor": 0.6, "storage_factor": 0.5, 
            "performance_headroom": 1.05, "reader_sizing_factor": 0.6
        },
        "DEV": {
            "cpu_factor": 0.25, "ram_factor": 0.4, "storage_factor": 0.3, 
            "performance_headroom": 1.0, "reader_sizing_factor": 0.4
        }
    }
    
    def __init__(self):
        self.aws_available = self._check_aws_credentials() if HAS_BOTO3 else False
        self.recommendations = {}
        self.inputs = {}
    
    def _check_aws_credentials(self):
        """Check if AWS credentials are available"""
        if not HAS_BOTO3:
            return False
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            return credentials is not None
        except Exception:
            return False
    
    def _get_instance_data(self, engine, region):
        """Get instance data - enhanced with reader/writer support"""
        instances = {
            "postgres": [
                {"type": "db.t3.micro", "vCPU": 2, "memory": 1, "pricing": {"ondemand": 0.026}},
                {"type": "db.t3.small", "vCPU": 2, "memory": 2, "pricing": {"ondemand": 0.051}},
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.102}},
                {"type": "db.t3.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.204}},
                {"type": "db.t3.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand": 0.408}},
                {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.192}},
                {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand": 0.384}},
                {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "pricing": {"ondemand": 0.768}},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.24}},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 0.48}},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand": 0.96}},
            ],
            "aurora-postgresql": [
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.082}},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.285}},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 0.57}},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand": 1.14}},
                {"type": "db.r6g.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.256}},
            ]
        }
        
        return instances.get(engine, instances["postgres"])
    
    def _parse_read_write_ratio(self):
        """Parse read/write ratio from inputs"""
        if self.inputs.get("workload_pattern") in self.WORKLOAD_PATTERNS:
            pattern = self.WORKLOAD_PATTERNS[self.inputs["workload_pattern"]]
            return pattern["read_percentage"], pattern["write_percentage"]
        
        # Parse custom ratio like "60:40"
        ratio_str = self.inputs.get("read_write_ratio", "60:40")
        try:
            read_str, write_str = ratio_str.split(":")
            read_pct = int(read_str.strip())
            write_pct = int(write_str.strip())
            
            # Normalize to 100%
            total = read_pct + write_pct
            if total != 100:
                read_pct = (read_pct / total) * 100
                write_pct = (write_pct / total) * 100
            
            return read_pct, write_pct
        except:
            return 60, 40
    
    def calculate_requirements(self, env):
        """Calculate requirements for a specific environment with reader/writer logic"""
        profile = self.ENV_PROFILES[env]
        deployment_config = self.DEPLOYMENT_OPTIONS[self.inputs["deployment"]]
        
        # Calculate base requirements
        base_vcpus = self.inputs["on_prem_cores"] * (self.inputs["peak_cpu_percent"] / 100)
        base_ram = self.inputs["on_prem_ram_gb"] * (self.inputs["peak_ram_percent"] / 100)
        
        # Apply environment-specific factors
        env_vcpus = base_vcpus * profile["cpu_factor"] * profile["performance_headroom"]
        env_ram = base_ram * profile["ram_factor"] * profile["performance_headroom"]
        
        # Calculate read/write workload distribution
        read_pct, write_pct = self._parse_read_write_ratio()
        
        # Calculate writer requirements (handles all writes + some reads)
        writer_cpu_requirement = env_vcpus * (write_pct / 100 + 0.3 * read_pct / 100)
        writer_ram_requirement = env_ram * (write_pct / 100 + 0.3 * read_pct / 100)
        
        # Environment minimums
        env_minimums = {
            "PROD": {"cpu": 4, "ram": 8},
            "SQA": {"cpu": 2, "ram": 4}, 
            "QA": {"cpu": 2, "ram": 4},
            "DEV": {"cpu": 1, "ram": 2}
        }
        
        min_reqs = env_minimums[env]
        final_writer_cpu = max(int(writer_cpu_requirement), min_reqs["cpu"])
        final_writer_ram = max(int(writer_ram_requirement), min_reqs["ram"])
        
        # Get available instances
        instances = self._get_instance_data(self.inputs["engine"], self.inputs["region"])
        
        # Select writer instance
        writer_instance = self._select_instance(final_writer_cpu, final_writer_ram, env, instances, "writer")
        
        # Calculate reader requirements if Multi-AZ deployment
        reader_recommendations = {}
        if deployment_config["has_readers"]:
            # Reader handles remaining read workload
            remaining_read_workload = 0.7 * read_pct / 100
            reader_cpu_per_instance = env_vcpus * remaining_read_workload / deployment_config["reader_count"]
            reader_ram_per_instance = env_ram * remaining_read_workload / deployment_config["reader_count"]
            
            # Apply reader sizing factor
            reader_cpu_per_instance *= profile["reader_sizing_factor"]
            reader_ram_per_instance *= profile["reader_sizing_factor"]
            
            # Apply minimums for readers
            reader_min_cpu = max(1, min_reqs["cpu"] // 2) if env != "PROD" else min_reqs["cpu"]
            reader_min_ram = max(1, min_reqs["ram"] // 2) if env != "PROD" else min_reqs["ram"]
            
            final_reader_cpu = max(int(reader_cpu_per_instance), reader_min_cpu)
            final_reader_ram = max(int(reader_ram_per_instance), reader_min_ram)
            
            # Select reader instance
            reader_instance = self._select_instance(final_reader_cpu, final_reader_ram, env, instances, "reader")
            
            reader_recommendations = {
                "instance_type": reader_instance["type"],
                "vCPUs": final_reader_cpu,
                "RAM_GB": final_reader_ram,
                "actual_vCPUs": reader_instance["vCPU"],
                "actual_RAM_GB": reader_instance["memory"],
                "count": deployment_config["reader_count"],
                "total_vCPUs": reader_instance["vCPU"] * deployment_config["reader_count"],
                "total_RAM_GB": reader_instance["memory"] * deployment_config["reader_count"],
                "instance_cost_per_reader": reader_instance["pricing"]["ondemand"] * 24 * 30,
                "total_reader_cost": reader_instance["pricing"]["ondemand"] * 24 * 30 * deployment_config["reader_count"]
            }
        
        # Calculate storage and costs
        storage = self._calculate_storage(env)
        costs = self._calculate_costs_with_readers(writer_instance, reader_recommendations, storage, env)
        
        # Generate advisories
        advisories = self._generate_advisories(writer_instance, reader_recommendations, env, read_pct, write_pct)
        
        return {
            "environment": env,
            "deployment_type": self.inputs["deployment"],
            "workload_pattern": f"{read_pct:.0f}% reads, {write_pct:.0f}% writes",
            
            # Writer information
            "writer": {
                "instance_type": writer_instance["type"],
                "vCPUs": final_writer_cpu,
                "RAM_GB": final_writer_ram,
                "actual_vCPUs": writer_instance["vCPU"],
                "actual_RAM_GB": writer_instance["memory"],
                "monthly_cost": writer_instance["pricing"]["ondemand"] * 24 * 30
            },
            
            # Reader information (if applicable)
            "readers": reader_recommendations if reader_recommendations else None,
            
            # Legacy fields for backward compatibility
            "instance_type": writer_instance["type"],
            "vCPUs": final_writer_cpu,
            "RAM_GB": final_writer_ram,
            "actual_vCPUs": writer_instance["vCPU"],
            "actual_RAM_GB": writer_instance["memory"],
            
            # Infrastructure details
            "storage_GB": storage,
            
            # Cost breakdown
            "instance_cost": writer_instance["pricing"]["ondemand"] * 24 * 30,
            "reader_cost": reader_recommendations.get("total_reader_cost", 0) if reader_recommendations else 0,
            "storage_cost": storage * 0.10,
            "total_cost": costs["total"],
            
            # Additional information
            "advisories": advisories,
            "has_readers": deployment_config["has_readers"]
        }
    
    def _select_instance(self, required_vcpus, required_ram, env, instances, instance_role="writer"):
        """Select optimal instance based on requirements and role"""
        suitable = []
        
        for instance in instances:
            if instance["vCPU"] >= required_vcpus and instance["memory"] >= required_ram:
                suitable.append(instance)
        
        if not suitable:
            # Relaxed matching for non-prod environments
            tolerance = 0.8 if env in ["DEV", "QA"] else 0.9
            if instance_role == "reader":
                tolerance *= 0.8
            
            for instance in instances:
                if (instance["vCPU"] >= required_vcpus * tolerance and 
                    instance["memory"] >= required_ram * tolerance):
                    suitable.append(instance)
        
        if not suitable:
            return instances[-1]  # Return largest if nothing fits
        
        # Selection strategy based on environment and role
        if env == "PROD" and instance_role == "writer":
            # Balance performance and cost for production writers
            def score(inst):
                headroom = (inst["vCPU"] + inst["memory"]) / (required_vcpus + required_ram)
                cost_factor = 1000 / (inst["pricing"]["ondemand"] + 1)
                return min(headroom, 2.0) * 0.7 + cost_factor * 0.3
            return max(suitable, key=score)
        else:
            # Cost-optimize for non-production or readers
            return min(suitable, key=lambda x: x["pricing"]["ondemand"])
    
    def _calculate_storage(self, env):
        """Calculate storage requirements"""
        profile = self.ENV_PROFILES[env]
        base_storage = self.inputs["storage_current_gb"]
        return max(20, int(base_storage * profile["storage_factor"] * 1.3))
    
    def _calculate_costs_with_readers(self, writer_instance, reader_recommendations, storage, env):
        """Calculate monthly costs including readers"""
        
        # Writer instance cost
        writer_monthly = writer_instance["pricing"]["ondemand"] * 24 * 30
        
        # Reader instance costs
        reader_monthly = 0
        if reader_recommendations:
            reader_monthly = reader_recommendations.get("total_reader_cost", 0)
        
        # Storage cost
        storage_monthly = storage * 0.10
        
        # Backup cost
        backup_monthly = storage * 0.095 * 0.25
        
        total_monthly = writer_monthly + reader_monthly + storage_monthly + backup_monthly
        
        return {
            "writer": writer_monthly,
            "reader": reader_monthly,
            "storage": storage_monthly,
            "backup": backup_monthly,
            "total": total_monthly
        }
    
    def _generate_advisories(self, writer_instance, reader_recommendations, env, read_pct, write_pct):
        """Generate optimization advisories"""
        advisories = []
        
        # Reader-specific advisories
        if reader_recommendations:
            reader_count = reader_recommendations["count"]
            
            if reader_count > 2 and env in ["DEV", "QA"]:
                advisories.append(f"💡 Consider reducing readers to 1 for {env} environment to save costs")
            
            if reader_recommendations["actual_vCPUs"] > writer_instance["vCPU"]:
                advisories.append("🔄 Reader instances are larger than writer - consider rebalancing")
        
        # Workload pattern advisories
        if read_pct > 80 and not reader_recommendations:
            advisories.append("📊 High read workload detected - consider adding read replicas")
        
        if write_pct > 70 and reader_recommendations and reader_recommendations["count"] > 1:
            advisories.append("✏️ Write-heavy workload detected - focus resources on writer instance")
        
        # Environment-specific advisories
        if env == "PROD":
            if self.inputs.get("deployment") == "Single-AZ":
                advisories.append("🚨 Production should use Multi-AZ deployment for high availability")
            
            if not reader_recommendations:
                advisories.append("📖 Consider adding read replicas to offload read traffic from the writer")
        
        elif env in ["DEV", "QA"]:
            total_cost = writer_instance["pricing"]["ondemand"] * 24 * 30
            if reader_recommendations:
                total_cost += reader_recommendations["total_reader_cost"]
            
            if total_cost > 500:
                advisories.append(f"💰 Consider smaller instances or Single-AZ deployment for {env} environment")
        
        # Aurora-specific advisories
        if "aurora" in self.inputs["engine"] and reader_recommendations:
            advisories.append("🚀 Consider Aurora Auto Scaling for readers based on CPU utilization")
        
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
    page_title="AWS RDS Sizing Tool with Reader/Writer Logic",
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
    .writer-box {
        background: #e3f2fd;
        border: 2px solid #2196f3;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .reader-box {
        background: #f3e5f5;
        border: 2px solid #9c27b0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .workload-info {
        background: #f8f9fa;
        border-left: 4px solid #6c757d;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize calculator
@st.cache_resource
def get_calculator():
    return RobustRDSSizingCalculator()

calculator = get_calculator()

# Header
st.title("🚀 AWS RDS & Aurora Sizing Tool")
st.markdown("**Enhanced with Reader/Writer sizing for Multi-AZ deployments**")

# Dependency status
col1, col2 = st.columns([3, 1])
with col1:
    deps_status = []
    if not HAS_PLOTLY:
        deps_status.append("📊 Plotly (for advanced charts)")
    if not HAS_BOTO3:
        deps_status.append("☁️ Boto3 (for real-time pricing)")
    
    if deps_status:
        st.warning(f"⚠️ Optional features unavailable: {', '.join(deps_status)}")
        st.info("💡 Install missing packages: pip install plotly boto3")
    else:
        st.success("✅ All dependencies available")

with col2:
    if st.button("📦 Install Dependencies"):
        st.code("pip install streamlit pandas plotly boto3 numpy")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # AWS Settings
    with st.expander("☁️ AWS Settings", expanded=True):
        region = st.selectbox("Region", ["us-east-1", "us-west-1", "us-west-2", "eu-west-1"], index=0)
        engine = st.selectbox("Database Engine", calculator.ENGINES, index=2)
        deployment = st.selectbox("Deployment", list(calculator.DEPLOYMENT_OPTIONS.keys()), index=1)
        
        # Show deployment info
        deployment_info = calculator.DEPLOYMENT_OPTIONS[deployment]
        st.info(f"📖 {deployment_info['description']}")
        if deployment_info['has_readers']:
            st.success(f"👥 Will include {deployment_info['reader_count']} reader instance(s)")
    
    # Workload Profile
    with st.expander("📊 Workload Profile", expanded=True):
        workload_pattern = st.selectbox(
            "Workload Pattern", 
            list(calculator.WORKLOAD_PATTERNS.keys()),
            index=0,
            format_func=lambda x: f"{x.replace('_', ' ').title()} ({calculator.WORKLOAD_PATTERNS[x]['read_percentage']}% reads)"
        )
        
        # Custom read/write ratio
        if workload_pattern == "MIXED":
            read_write_ratio = st.text_input("Custom Read:Write Ratio", value="50:50")
        else:
            pattern_info = calculator.WORKLOAD_PATTERNS[workload_pattern]
            read_write_ratio = f"{pattern_info['read_percentage']}:{pattern_info['write_percentage']}"
            st.info(f"📈 {pattern_info['description']}: {read_write_ratio}")
    
    # Current Workload
    with st.expander("🖥️ Current Workload", expanded=True):
        cores = st.number_input("CPU Cores", min_value=1, max_value=128, value=8, step=1)
        cpu_util = st.slider("Peak CPU %", 1, 100, 70)
        ram = st.number_input("RAM (GB)", min_value=1, max_value=1024, value=32, step=1)
        ram_util = st.slider("Peak RAM %", 1, 100, 80)
    
    with st.expander("💾 Storage", expanded=True):
        storage = st.number_input("Current Storage (GB)", min_value=1, value=250)
        growth = st.number_input("Annual Growth %", min_value=0, max_value=100, value=20)

# Update calculator inputs
calculator.inputs = {
    "region": region,
    "engine": engine,
    "deployment": deployment,
    "workload_pattern": workload_pattern,
    "read_write_ratio": read_write_ratio,
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
    if st.button("🚀 Generate Reader/Writer Sizing", type="primary", use_container_width=True):
        with st.spinner("🔄 Calculating reader and writer sizing..."):
            start_time = time.time()
            
            try:
                results = calculator.generate_all_recommendations()
                st.session_state['results'] = results
                st.session_state['generation_time'] = time.time() - start_time
                
                # Check if deployment has readers
                deployment_config = calculator.DEPLOYMENT_OPTIONS[deployment]
                if deployment_config['has_readers']:
                    st.success(f"✅ Generated recommendations with {deployment_config['reader_count']} reader instance(s)")
                else:
                    st.success("✅ Generated Single-AZ recommendations (no readers)")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

with col2:
    if st.button("📊 Export CSV", use_container_width=True):
        if 'results' in st.session_state:
            # Simple CSV export
            results = st.session_state['results']
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            
            if valid_results:
                export_data = []
                for env, result in valid_results.items():
                    row = {
                        'Environment': env,
                        'Deployment': deployment,
                        'Writer Instance': result['writer']['instance_type'],
                        'Writer Cost': result['instance_cost'],
                        'Reader Instance': result['readers']['instance_type'] if result.get('readers') else 'None',
                        'Reader Count': result['readers']['count'] if result.get('readers') else 0,
                        'Reader Cost': result.get('reader_cost', 0),
                        'Total Cost': result['total_cost']
                    }
                    export_data.append(row)
                
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"rds_reader_writer_sizing_{int(time.time())}.csv",
                    mime="text/csv"
                )

with col3:
    if st.button("🔧 Debug Info", use_container_width=True):
        st.session_state['show_debug'] = not st.session_state.get('show_debug', False)

# Display Results
if 'results' in st.session_state:
    results = st.session_state['results']
    
    # Workload Pattern Summary
    deployment_config = calculator.DEPLOYMENT_OPTIONS[deployment]
    
    st.markdown(f"""
    <div class="workload-info">
        <h4>🎯 Workload Analysis</h4>
        <p><strong>Deployment:</strong> {deployment} ({deployment_config['description']})</p>
        <p><strong>Workload Pattern:</strong> {workload_pattern.replace('_', ' ').title()}</p>
        <p><strong>Read/Write Distribution:</strong> {read_write_ratio}</p>
        {f"<p><strong>Reader Instances:</strong> {deployment_config['reader_count']} per environment</p>" if deployment_config['has_readers'] else "<p><strong>Reader Instances:</strong> None (Single-AZ deployment)</p>"}
    </div>
    """, unsafe_allow_html=True)
    
    # Summary Metrics
    st.header("📊 Recommendation Summary")
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        prod_result = valid_results.get('PROD', list(valid_results.values())[0])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{prod_result['writer']['instance_type']}</div>
                <div class="metric-label">Production Writer</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if prod_result.get('readers'):
                reader_info = f"{prod_result['readers']['instance_type']} x{prod_result['readers']['count']}"
            else:
                reader_info = "None"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value" style="font-size: 1.5rem;">{reader_info}</div>
                <div class="metric-label">Production Readers</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">${prod_result['total_cost']:,.0f}</div>
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
        
        # Enhanced Results Display with Reader/Writer Breakdown
        st.header("📋 Environment-Specific Reader/Writer Recommendations")
        
        for env, result in valid_results.items():
            with st.expander(f"{env} Environment - ${result['total_cost']:,.2f}/month", expanded=True):
                
                col1, col2 = st.columns(2)
                
                # Writer Information
                with col1:
                    writer = result['writer']
                    st.markdown(f"""
                    <div class="writer-box">
                        <h4>✍️ Writer Instance</h4>
                        <p><strong>Instance:</strong> {writer['instance_type']}</p>
                        <p><strong>vCPUs:</strong> {writer['actual_vCPUs']} (required: {writer['vCPUs']})</p>
                        <p><strong>RAM:</strong> {writer['actual_RAM_GB']}GB (required: {writer['RAM_GB']}GB)</p>
                        <p><strong>Monthly Cost:</strong> ${writer['monthly_cost']:,.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Reader Information
                with col2:
                    if result.get('readers'):
                        readers = result['readers']
                        st.markdown(f"""
                        <div class="reader-box">
                            <h4>📖 Reader Instances</h4>
                            <p><strong>Instance:</strong> {readers['instance_type']} x{readers['count']}</p>
                            <p><strong>vCPUs per reader:</strong> {readers['actual_vCPUs']} (required: {readers['vCPUs']})</p>
                            <p><strong>RAM per reader:</strong> {readers['actual_RAM_GB']}GB (required: {readers['RAM_GB']}GB)</p>
                            <p><strong>Total vCPUs:</strong> {readers['total_vCPUs']}</p>
                            <p><strong>Total RAM:</strong> {readers['total_RAM_GB']}GB</p>
                            <p><strong>Monthly Cost:</strong> ${readers['total_reader_cost']:,.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="reader-box">
                            <h4>📖 Reader Instances</h4>
                            <p><strong>No readers</strong> (Single-AZ deployment)</p>
                            <p>All read and write operations handled by the writer instance</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Cost Breakdown
                st.subheader("💰 Cost Breakdown")
                cost_cols = st.columns(4)
                
                with cost_cols[0]:
                    st.metric("Writer", f"${result['instance_cost']:,.2f}")
                with cost_cols[1]:
                    st.metric("Readers", f"${result.get('reader_cost', 0):,.2f}")
                with cost_cols[2]:
                    st.metric("Storage", f"${result['storage_cost']:,.2f}")
                with cost_cols[3]:
                    st.metric("Total", f"${result['total_cost']:,.2f}")
                
                # Workload Pattern for this environment
                st.info(f"🎯 **Workload Pattern:** {result['workload_pattern']}")
                
                # Advisories for this environment
                if result.get('advisories'):
                    st.subheader("💡 Optimization Advisories")
                    for advisory in result['advisories']:
                        st.info(advisory)
        
        # Simple Charts (if Plotly available) or Tables
        st.header("📈 Cost Analysis")
        
        if HAS_PLOTLY:
            # Create Plotly charts
            chart_data = []
            for env, result in valid_results.items():
                chart_data.append({
                    'Environment': env,
                    'Writer Cost': result['instance_cost'],
                    'Reader Cost': result.get('reader_cost', 0),
                    'Storage Cost': result['storage_cost'],
                    'Total Cost': result['total_cost']
                })
            
            df = pd.DataFrame(chart_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stacked bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Writer',
                    x=df['Environment'],
                    y=df['Writer Cost'],
                    marker_color='#2196F3'
                ))
                
                fig.add_trace(go.Bar(
                    name='Readers',
                    x=df['Environment'],
                    y=df['Reader Cost'],
                    marker_color='#9C27B0'
                ))
                
                fig.add_trace(go.Bar(
                    name='Storage',
                    x=df['Environment'],
                    y=df['Storage Cost'],
                    marker_color='#4CAF50'
                ))
                
                fig.update_layout(
                    title='Cost Breakdown by Environment',
                    barmode='stack',
                    xaxis_title='Environment',
                    yaxis_title='Monthly Cost ($)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Total cost comparison
                fig2 = px.bar(df, x='Environment', y='Total Cost', 
                             title='Total Monthly Cost by Environment',
                             color='Environment')
                st.plotly_chart(fig2, use_container_width=True)
        
        else:
            # Fallback to simple table
            st.subheader("Cost Comparison Table")
            
            table_data = []
            for env, result in valid_results.items():
                table_data.append({
                    'Environment': env,
                    'Writer Cost': f"${result['instance_cost']:,.2f}",
                    'Reader Cost': f"${result.get('reader_cost', 0):,.2f}",
                    'Storage Cost': f"${result['storage_cost']:,.2f}",
                    'Total Cost': f"${result['total_cost']:,.2f}"
                })
            
            st.table(pd.DataFrame(table_data))
        
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
    
    with st.expander("Deployment Configuration"):
        st.json(calculator.DEPLOYMENT_OPTIONS[deployment])
    
    if 'results' in st.session_state:
        with st.expander("Raw Results"):
            st.json(st.session_state['results'])

# Footer
st.markdown("---")
st.markdown("""
**🎯 Reader/Writer Features:**
- ✅ **Separate Writer/Reader Sizing**: Different instances optimized for read vs write workloads
- ✅ **Workload Pattern Analysis**: Intelligent distribution based on read/write ratios
- ✅ **Environment-Specific Scaling**: Readers sized appropriately for each environment
- ✅ **Cost Optimization**: Smaller readers for development environments
- ✅ **Multi-AZ Support**: Proper handling of different deployment configurations

**📦 Dependencies:** This app works best with all dependencies installed. Missing packages reduce functionality but core features remain available.
""")