import streamlit as st
import pandas as pd
import time
import traceback
import io

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

class EnhancedDemoRDSSizingCalculator:
    ENGINES = ['oracle-ee', 'oracle-se', 'postgres', 'aurora-postgresql', 'aurora-mysql', 'sqlserver']
    
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
    
    def refresh_aws_credentials(self):
        """Refresh AWS credentials and return status"""
        if not HAS_BOTO3:
            self.aws_available = False
            return False
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            self.aws_available = credentials is not None
            return self.aws_available
        except Exception as e:
            self.aws_available = False
            return False
    
    def _get_instance_data(self, engine, region):
        """Get instance data - enhanced with more variety"""
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
                {"type": "db.m5.4xlarge", "vCPU": 16, "memory": 64, "pricing": {"ondemand": 1.536}},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.24}},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 0.48}},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand": 0.96}},
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
        
        # Fallback to postgres for other engines for demo purposes
        default_instances = instances["postgres"]
        for eng in self.ENGINES:
            if eng not in instances:
                instances[eng] = default_instances
        
        return instances.get(engine, default_instances)
    
    def _parse_read_write_ratio(self):
        """Parse read/write ratio from inputs"""
        if self.inputs.get("workload_pattern") in self.WORKLOAD_PATTERNS:
            pattern = self.WORKLOAD_PATTERNS[self.inputs["workload_pattern"]]
            # For non-mixed patterns, always use the defined ratio
            if self.inputs["workload_pattern"] != 'MIXED':
                return pattern["read_percentage"], pattern["write_percentage"]

        # For 'MIXED' or custom inputs, parse the ratio string
        ratio_str = self.inputs.get("read_write_ratio", "60:40")
        try:
            read_str, write_str = ratio_str.split(":")
            read_pct = int(read_str.strip())
            write_pct = int(write_str.strip())
            
            # Normalize to 100%
            total = read_pct + write_pct
            if total > 0 and total != 100:
                read_pct = (read_pct / total) * 100
                write_pct = (write_pct / total) * 100
            
            return read_pct, write_pct
        except:
            return 60, 40 # Fallback
    
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
            
            "writer": {
                "instance_type": writer_instance["type"],
                "vCPUs": final_writer_cpu,
                "RAM_GB": final_writer_ram,
                "actual_vCPUs": writer_instance["vCPU"],
                "actual_RAM_GB": writer_instance["memory"],
                "monthly_cost": writer_instance["pricing"]["ondemand"] * 24 * 30
            },
            
            "readers": reader_recommendations,
            
            "instance_type": writer_instance["type"],
            "vCPUs": final_writer_cpu,
            "RAM_GB": final_writer_ram,
            "actual_vCPUs": writer_instance["vCPU"],
            "actual_RAM_GB": writer_instance["memory"],
            
            "storage_GB": storage,
            
            "instance_cost": costs["writer"],
            "reader_cost": costs.get("reader", 0),
            "storage_cost": costs["storage"],
            "total_cost": costs["total"],
            
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
            tolerance = 0.8 if env in ["DEV", "QA"] else 0.9
            
            if instance_role == "reader":
                tolerance *= 0.8
            
            for instance in instances:
                if (instance["vCPU"] >= required_vcpus * tolerance and 
                    instance["memory"] >= required_ram * tolerance):
                    suitable.append(instance)
        
        if not suitable:
            return instances[-1]  # Return largest if nothing fits
        
        if env == "PROD" and instance_role == "writer":
            def score(inst):
                headroom = (inst["vCPU"] + inst["memory"]) / (required_vcpus + required_ram)
                cost_factor = 1000 / (inst["pricing"]["ondemand"] + 1)
                return min(headroom, 2.0) * 0.7 + cost_factor * 0.3
            return max(suitable, key=score)
        else:
            return min(suitable, key=lambda x: x["pricing"]["ondemand"])
    
    def _calculate_storage(self, env):
        """Calculate storage requirements"""
        profile = self.ENV_PROFILES[env]
        base_storage = self.inputs["storage_current_gb"]
        return max(20, int(base_storage * profile["storage_factor"] * 1.3))
    
    def _calculate_costs_with_readers(self, writer_instance, reader_recommendations, storage, env):
        """Calculate monthly costs including readers"""
        writer_monthly = writer_instance["pricing"]["ondemand"] * 24 * 30
        
        reader_monthly = 0
        if reader_recommendations:
            reader_monthly = reader_recommendations.get("total_reader_cost", 0)
        
        storage_monthly = storage * 0.10
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
        
        if reader_recommendations:
            reader_count = reader_recommendations["count"]
            if reader_count > 2 and env in ["DEV", "QA"]:
                advisories.append(f"💡 Consider reducing readers to 1 for {env} environment to save costs")
            if reader_recommendations["actual_vCPUs"] > writer_instance["vCPU"]:
                advisories.append("🔄 Reader instances are larger than writer - consider rebalancing")
        
        if read_pct > 80 and not reader_recommendations:
            advisories.append("📊 High read workload detected - consider adding read replicas")
        if write_pct > 70 and reader_recommendations and reader_recommendations["count"] > 1:
            advisories.append("✏️ Write-heavy workload detected - focus resources on writer instance")
        
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
st.set_page_config(page_title="Enhanced AWS RDS Sizing Tool", layout="wide", page_icon="🚀")

# Custom CSS
st.markdown("""
<style>
    /* [Existing CSS styles remain unchanged] */
    .main > div { padding-top: 2rem; }
    .metric-container { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; text-align: center; margin: 0.5rem 0; }
    .metric-value { font-size: 2rem; font-weight: bold; color: #111; }
    .metric-label { font-size: 0.9rem; opacity: 0.9; color: #eee; }
    .writer-box { background: #e3f2fd; border: 2px solid #2196f3; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    .reader-box { background: #f3e5f5; border: 2px solid #9c27b0; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    .advisory-box { background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; }
    .status-good { background: #d4edda; border-left: 4px solid #28a745; padding: 0.75rem; margin: 0.5rem 0; color: #155724; }
    .status-warning { background: #fff3cd; border-left: 4px solid #ffc107; padding: 0.75rem; margin: 0.5rem 0; color: #856404; }
    .workload-info { background: #f8f9fa; border-left: 4px solid #6c757d; padding: 1rem; margin: 1rem 0; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# Initialize calculator
@st.cache_resource
def get_calculator():
    return EnhancedDemoRDSSizingCalculator()

calculator = get_calculator()

# --- HEADER ---
st.title("🚀 Enhanced AWS RDS & Aurora Sizing Tool")
st.markdown("**Real-time AWS pricing with separate Reader/Writer sizing and Bulk Upload capability**")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    input_mode = st.radio("Input Mode", ["Manual Input", "Bulk Upload (CSV/Excel)"], horizontal=True)
    
    # --- MANUAL INPUT MODE ---
    if input_mode == "Manual Input":
        with st.expander("☁️ AWS Settings", expanded=True):
            region = st.selectbox("Region", ["us-east-1", "us-west-1", "us-west-2", "eu-west-1"], index=0, key="manual_region")
            engine = st.selectbox("Database Engine", calculator.ENGINES, index=2, key="manual_engine")
            deployment = st.selectbox("Deployment", list(calculator.DEPLOYMENT_OPTIONS.keys()), index=1, key="manual_deployment")
            deployment_info = calculator.DEPLOYMENT_OPTIONS[deployment]
            st.info(f"📖 {deployment_info['description']}")
        
        with st.expander("📊 Workload Profile", expanded=True):
            workload_pattern = st.selectbox("Workload Pattern", list(calculator.WORKLOAD_PATTERNS.keys()), index=0,
                                            format_func=lambda x: f"{x.replace('_', ' ').title()} ({calculator.WORKLOAD_PATTERNS[x]['read_percentage']}% reads)",
                                            key="manual_workload")
            if workload_pattern == "MIXED":
                read_write_ratio = st.text_input("Custom Read:Write Ratio", value="50:50", help="e.g., 70:30", key="manual_ratio")
            else:
                pattern_info = calculator.WORKLOAD_PATTERNS[workload_pattern]
                read_write_ratio = f"{pattern_info['read_percentage']}:{pattern_info['write_percentage']}"
                st.info(f"📈 {pattern_info['description']}: {read_write_ratio}")

        with st.expander("🖥️ Current Workload", expanded=True):
            cores = st.number_input("CPU Cores", min_value=1, max_value=128, value=8, step=1, key="manual_cores")
            cpu_util = st.slider("Peak CPU %", 1, 100, 70, key="manual_cpu_util")
            ram = st.number_input("RAM (GB)", min_value=1, max_value=1024, value=32, step=1, key="manual_ram")
            ram_util = st.slider("Peak RAM %", 1, 100, 80, key="manual_ram_util")
        
        with st.expander("💾 Storage", expanded=True):
            storage = st.number_input("Current Storage (GB)", min_value=1, value=250, key="manual_storage")
            growth = st.number_input("Annual Growth %", min_value=0, max_value=100, value=20, key="manual_growth")

        # Set calculator inputs for manual mode
        calculator.inputs = {
            "region": region, "engine": engine, "deployment": deployment, "workload_pattern": workload_pattern,
            "read_write_ratio": read_write_ratio, "on_prem_cores": cores, "peak_cpu_percent": cpu_util,
            "on_prem_ram_gb": ram, "peak_ram_percent": ram_util, "storage_current_gb": storage,
            "storage_growth_rate": growth/100
        }

    # --- BULK UPLOAD MODE ---
    else:
        st.info("Upload a CSV or Excel file with your workload scenarios.")
        uploaded_file = st.file_uploader("Upload Scenarios File", type=["csv", "xlsx"])
        
        st.markdown("---")
        st.markdown("##### Download Template")
        template_df = pd.DataFrame([{
            "scenario_name": "My Web App", "region": "us-east-1", "engine": "postgres",
            "deployment": "Multi-AZ", "workload_pattern": "OLTP_BALANCED", "read_write_ratio": "60:40",
            "on_prem_cores": 8, "peak_cpu_percent": 70, "on_prem_ram_gb": 32, "peak_ram_percent": 80,
            "storage_current_gb": 250, "storage_growth_rate_percent": 20
        }])
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download CSV Template", data=csv_template, file_name="sizing_template.csv", mime="text/csv")


# --- MAIN CONTENT & ACTIONS ---
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    if st.button("🚀 Generate Sizing", type="primary", use_container_width=True):
        # Clear previous results to avoid mixing displays
        for key in ['results', 'bulk_results', 'generation_time']:
            if key in st.session_state:
                del st.session_state[key]
        
        start_time = time.time()
        
        # --- MANUAL MODE LOGIC ---
        if input_mode == "Manual Input":
            with st.spinner("🔄 Calculating sizing..."):
                try:
                    results = calculator.generate_all_recommendations()
                    st.session_state['results'] = results
                    st.session_state['generation_time'] = time.time() - start_time
                    st.success("✅ Generated recommendations successfully!")
                except Exception as e:
                    st.error(f"❌ Error during calculation: {e}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

        # --- BULK MODE LOGIC ---
        else:
            if uploaded_file is None:
                st.error("Please upload a file to proceed with bulk sizing.")
            else:
                with st.spinner(f"Reading and processing '{uploaded_file.name}'..."):
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)

                        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('%', 'percent')
                        bulk_results = []
                        total_rows = len(df)
                        progress_bar = st.progress(0, text=f"Processing {total_rows} scenarios...")

                        for index, row in df.iterrows():
                            progress_bar.progress((index + 1) / total_rows, text=f"Processing scenario: {row.get('scenario_name', index + 1)}")
                            
                            calc_inputs = {
                                "scenario_name": row.get("scenario_name", f"Scenario {index + 1}"),
                                "region": row.get("region", "us-east-1"),
                                "engine": row.get("engine", "postgres"),
                                "deployment": row.get("deployment", "Multi-AZ"),
                                "workload_pattern": row.get("workload_pattern", "MIXED"),
                                "read_write_ratio": str(row.get("read_write_ratio", "50:50")),
                                "on_prem_cores": int(row.get("on_prem_cores", 8)),
                                "peak_cpu_percent": int(row.get("peak_cpu_percent", 70)),
                                "on_prem_ram_gb": int(row.get("on_prem_ram_gb", 32)),
                                "peak_ram_percent": int(row.get("peak_ram_percent", 80)),
                                "storage_current_gb": int(row.get("storage_current_gb", 250)),
                                "storage_growth_rate": float(row.get("storage_growth_rate_percent", 20)) / 100
                            }
                            calculator.inputs = calc_inputs
                            recommendations = calculator.generate_all_recommendations()
                            bulk_results.append({"inputs": calc_inputs, "outputs": recommendations})
                        
                        st.session_state['bulk_results'] = bulk_results
                        st.session_state['generation_time'] = time.time() - start_time
                        progress_bar.empty()
                        st.success(f"✅ Processed {total_rows} scenarios successfully!")

                    except Exception as e:
                        st.error(f"❌ Error processing file: {e}")
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())

with col2:
    # The export button is now outside the generate button
    export_placeholder = st.empty()

with col3:
    if st.button("🔧 Debug Info", use_container_width=True):
        st.session_state['show_debug'] = not st.session_state.get('show_debug', False)

# --- RESULTS DISPLAY ---
def display_scenario_results(scenario_name, results, inputs, is_expanded=False):
    """Renders the complete output for a single scenario."""
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if not valid_results:
        st.error(f"Could not generate valid recommendations for scenario: **{scenario_name}**")
        return

    deployment_config = calculator.DEPLOYMENT_OPTIONS[inputs['deployment']]
    
    with st.expander(f"#### Scenario: {scenario_name} | {inputs['deployment']} | PROD Cost: ${valid_results.get('PROD', {}).get('total_cost', 0):,.0f}/month", expanded=is_expanded):
        
        # --- WORKLOAD ANALYSIS ---
        st.markdown(f"""
        <div class="workload-info">
            <p><strong>Workload Pattern:</strong> {inputs['workload_pattern'].replace('_', ' ').title()} | <strong>Read/Write Distribution:</strong> {inputs['read_write_ratio']}</p>
            {f"<p><strong>Reader Instances:</strong> {deployment_config['reader_count']} per environment</p>" if deployment_config['has_readers'] else "<p><strong>Reader Instances:</strong> None (Single-AZ deployment)</p>"}
        </div>
        """, unsafe_allow_html=True)
        
        # --- ENVIRONMENT DETAILS ---
        st.subheader("📋 Environment-Specific Recommendations")
        for env, result in valid_results.items():
            with st.container(border=True):
                st.markdown(f"##### {env} Environment - `${result['total_cost']:,.2f}/month`")
                col1, col2 = st.columns(2)
                
                with col1: # Writer
                    writer = result['writer']
                    st.markdown(f"""
                    <div class="writer-box">
                        <strong>✍️ Writer Instance</strong><br>
                        <strong>Instance:</strong> {writer['instance_type']}<br>
                        <strong>vCPUs:</strong> {writer['actual_vCPUs']} (req: {writer['vCPUs']}) | <strong>RAM:</strong> {writer['actual_RAM_GB']}GB (req: {writer['RAM_GB']}GB)<br>
                        <strong>Monthly Cost:</strong> ${writer['monthly_cost']:,.2f}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2: # Reader
                    if result.get('readers'):
                        readers = result['readers']
                        st.markdown(f"""
                        <div class="reader-box">
                            <strong>📖 Reader Instances</strong><br>
                            <strong>Instance:</strong> {readers['instance_type']} x{readers['count']}<br>
                            <strong>vCPUs/RAM per reader:</strong> {readers['actual_vCPUs']} / {readers['actual_RAM_GB']}GB<br>
                            <strong>Monthly Cost:</strong> ${readers['total_reader_cost']:,.2f}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="reader-box"><strong>📖 No reader instances</strong><br>Single-AZ deployment handles all traffic on the writer.</div>', unsafe_allow_html=True)
                
                if result.get('advisories'):
                    with st.container():
                        st.markdown("💡 **Advisories**")
                        for advisory in result['advisories']:
                            st.info(advisory, icon="💡")

        # --- CHARTS ---
        st.subheader("📈 Cost & Resource Analysis")
        chart_data = [{'Environment': env, 'Writer Cost': res['instance_cost'], 'Reader Cost': res.get('reader_cost', 0), 'Storage Cost': res['storage_cost']} for env, res in valid_results.items()]
        df = pd.DataFrame(chart_data)

        if HAS_PLOTLY:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(df, x='Environment', y=['Writer Cost', 'Reader Cost', 'Storage Cost'], title='Cost Breakdown by Environment', labels={'value': 'Monthly Cost ($)', 'variable': 'Cost Type'}, height=400)
                fig.update_layout(barmode='stack', legend_title_text='')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                resource_data = [{'env': env, 'type': 'Writer', 'vCPUs': res['writer']['actual_vCPUs'], 'RAM': res['writer']['actual_RAM_GB']} for env, res in valid_results.items()]
                if deployment_config['has_readers']:
                     resource_data.extend([{'env': env, 'type': 'Readers', 'vCPUs': res['readers']['total_vCPUs'], 'RAM': res['readers']['total_RAM_GB']} for env, res in valid_results.items()])
                res_df = pd.DataFrame(resource_data)
                fig_res = px.bar(res_df, x='env', y='vCPUs', color='type', title='vCPU Allocation: Writer vs Readers', labels={'env': 'Environment'}, height=400, barmode='group')
                st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.bar_chart(df.set_index('Environment'))

# --- RENDER RESULTS ---
# Manual results
if 'results' in st.session_state:
    st.header("📊 Recommendation Summary")
    display_scenario_results("Manual Input Scenario", st.session_state['results'], calculator.inputs, is_expanded=True)

# Bulk results
if 'bulk_results' in st.session_state:
    st.header("📊 Bulk Sizing Results")
    st.info(f"Showing results for {len(st.session_state['bulk_results'])} scenarios. Expand each section for details.")
    for i, scenario in enumerate(st.session_state['bulk_results']):
        display_scenario_results(scenario['inputs']['scenario_name'], scenario['outputs'], scenario['inputs'], is_expanded=(i == 0))
    
    # Aggregated charts for bulk results
    st.header("📈 Aggregated Analysis (All Scenarios)")
    agg_data = []
    for scenario in st.session_state['bulk_results']:
        prod_cost = scenario['outputs'].get('PROD', {}).get('total_cost', 0)
        if prod_cost > 0:
            agg_data.append({'Scenario': scenario['inputs']['scenario_name'], 'PROD Total Cost': prod_cost})
    
    if agg_data:
        agg_df = pd.DataFrame(agg_data)
        if HAS_PLOTLY:
            fig_agg = px.bar(agg_df, x='Scenario', y='PROD Total Cost', title='PROD Environment Monthly Cost Comparison', height=450)
            st.plotly_chart(fig_agg, use_container_width=True)
        else:
            st.bar_chart(agg_df.set_index('Scenario'))


# --- EXPORT LOGIC ---
if 'results' in st.session_state or 'bulk_results' in st.session_state:
    if export_placeholder.button("📊 Export Results", use_container_width=True):
        export_data = []
        scenarios_to_export = []

        if 'results' in st.session_state: # Manual mode
            scenarios_to_export.append({"inputs": calculator.inputs, "outputs": st.session_state['results']})
        elif 'bulk_results' in st.session_state: # Bulk mode
            scenarios_to_export = st.session_state['bulk_results']
        
        for scenario in scenarios_to_export:
            inputs = scenario['inputs']
            valid_results = {k: v for k, v in scenario['outputs'].items() if 'error' not in v}
            for env, result in valid_results.items():
                row = {
                    'Scenario Name': inputs.get('scenario_name', 'Manual Input'), 'Environment': env,
                    'Deployment': inputs['deployment'], 'Workload Pattern': result['workload_pattern'],
                    'Writer Instance': result['writer']['instance_type'], 'Writer vCPUs': result['writer']['actual_vCPUs'],
                    'Writer RAM (GB)': result['writer']['actual_RAM_GB'], 'Writer Cost': result['instance_cost'],
                }
                if result.get('readers'):
                    readers = result['readers']
                    row.update({'Reader Instance': readers['instance_type'], 'Reader Count': readers['count'], 'Reader vCPUs (each)': readers['actual_vCPUs'], 'Reader Cost (total)': readers['total_reader_cost']})
                else:
                    row.update({'Reader Instance': 'None', 'Reader Count': 0, 'Reader vCPUs (each)': 0, 'Reader Cost (total)': 0})
                
                row.update({'Storage (GB)': result['storage_GB'], 'Storage Cost': result['storage_cost'], 'Total Cost': result['total_cost']})
                export_data.append(row)
        
        if export_data:
            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Sizing CSV", data=csv,
                file_name=f"rds_sizing_{int(time.time())}.csv", mime="text/csv",
                key="download_button"
            )

# --- DEBUG INFO ---
if st.session_state.get('show_debug', False):
    st.header("🔧 Debug Information")
    with st.expander("Last Run Inputs (Manual Mode)"):
        st.json(calculator.inputs if input_mode == 'Manual Input' else "N/A: Bulk Upload Mode Active")
    if 'results' in st.session_state:
        with st.expander("Raw Manual Results"):
            st.json(st.session_state['results'])
    if 'bulk_results' in st.session_state:
        with st.expander("Raw Bulk Results"):
            st.json(st.session_state['bulk_results'])

# --- FOOTER ---
st.markdown("---")
st.markdown("""
**🎯 Key Features:**
- ✅ **Bulk Upload**: Size multiple scenarios at once by uploading a CSV or Excel file.
- ✅ **Reader/Writer Separation**: Separate sizing for writer and reader instances in Multi-AZ deployments.
- ✅ **Workload Pattern Analysis**: Different read/write ratios affect writer vs reader sizing.
- ✅ **Cost Optimization**: Reader instances optimized for cost in non-production environments.
- ✅ **Enhanced Visualizations**: Charts showing writer vs reader cost and resource distribution.
""")