import streamlit as st
import pandas as pd
import plotly.express as px
from rds_sizing import RDSDatabaseSizingCalculator
import time

# Configure enterprise-grade UI
st.set_page_config(
    page_title="Enterprise RDS/Aurora Sizing",
    layout="wide",
    page_icon=":bar_chart:"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    :root {
        --primary: #2563EB;
        --secondary: #1E40AF;
        --accent: #3B82F6;
        --light: #EFF6FF;
        --dark: #1F2937;
    }
    
    .reportview-container {
        background: #f8fafc;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, var(--primary), var(--secondary));
        color: white;
        padding-top: 2rem;
    }
    
    .stButton>button {
        background: var(--accent);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background: var(--secondary);
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 4px solid var(--accent);
    }
    
    .metric-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--dark);
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    .advisory-card {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .risk-matrix {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 8px;
        margin-top: 1rem;
    }
    
    .risk-cell {
        padding: 1rem;
        text-align: center;
        border-radius: 4px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize calculator
calculator = RDSDatabaseSizingCalculator()

# App header
st.title("Enterprise AWS RDS & Aurora Sizing Tool")
st.markdown("""
**Comprehensive migration planning for Oracle, PostgreSQL, Aurora with TCO analysis and optimization recommendations**  
*Enterprise-grade solution with real-time AWS pricing and risk assessment*
""")

# Sidebar configuration
with st.sidebar:
    st.header(":cloud: AWS Configuration")
    region = st.selectbox("AWS Region", ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "ap-southeast-1"], index=0)
    
    st.header(":gear: Database Settings")
    engine = st.selectbox("Database Engine", calculator.ENGINES, index=0)
    
    # Deployment model selection for supported engines
    supported_serverless = ['aurora-postgresql', 'aurora-mysql']
    if engine in supported_serverless:
        deployment_model = st.radio("Deployment Model", ["Provisioned", "Serverless"])
    else:
        deployment_model = "Provisioned"
    
    deployment = st.selectbox("Deployment Type", list(calculator.DEPLOYMENT_OPTIONS.keys()), index=1)
    storage_type = st.selectbox("Storage Type", list(calculator.STORAGE_TYPES.keys()), index=1)
    
    st.header(":chart_with_upwards_trend: Workload Profile")
    with st.expander("Compute Resources", expanded=True):
        cores = st.number_input("CPU Cores", min_value=1, value=16)
        cpu_util = st.slider("Peak CPU Utilization (%)", 1, 100, 65)
        ram = st.number_input("RAM (GB)", min_value=1, value=64)
        ram_util = st.slider("Peak RAM Utilization (%)", 1, 100, 75)
    
    with st.expander("Storage Configuration"):
        storage = st.number_input("Current Storage (GB)", min_value=1, value=500)
        growth = st.number_input("Annual Growth Rate (%)", min_value=0, max_value=100, value=15)
        iops = st.number_input("Peak IOPS", min_value=100, value=8000, step=1000)
        throughput = st.number_input("Peak Throughput (MB/s)", min_value=1, value=400)
    
    with st.expander("High Availability"):
        replicas = st.number_input("Read Replicas", min_value=0, max_value=15, value=1)
        backup_days = st.slider("Backup Retention (Days)", 0, 35, 7)
    
    with st.expander("Cost Optimization"):
        ri_term = st.selectbox("Reserved Instance Term", list(calculator.RI_DISCOUNTS.keys()), index=0)
        ri_duration = st.radio("RI Duration", ["1yr", "3yr"], index=0)
    
    with st.expander("Security & Compliance"):
        enable_encryption = st.checkbox("Encryption at Rest", True)
        enable_perf_insights = st.checkbox("Performance Insights", True)
    
    st.header(":moneybag: Financials")
    years = st.slider("Projection Years", 1, 5, 3)
    data_transfer = st.number_input("Monthly Data Transfer (GB)", min_value=0, value=100)

# Update calculator inputs
calculator.inputs.update({
    "region": region,
    "engine": engine,
    "deployment": deployment,
    "deployment_model": deployment_model,
    "storage_type": storage_type,
    "on_prem_cores": cores,
    "peak_cpu_percent": cpu_util,
    "on_prem_ram_gb": ram,
    "peak_ram_percent": ram_util,
    "storage_current_gb": storage,
    "storage_growth_rate": growth/100,
    "peak_iops": iops,
    "peak_throughput_mbps": throughput,
    "years": years,
    "ha_replicas": replicas,
    "backup_retention": backup_days,
    "enable_encryption": enable_encryption,
    "enable_perf_insights": enable_perf_insights,
    "monthly_data_transfer_gb": data_transfer,
    "ri_term": ri_term,
    "ri_duration": ri_duration
})

# Main dashboard
if st.button("Generate Sizing Recommendations", type="primary", use_container_width=True):
    start_time = time.time()
    
    with st.spinner("🚀 Generating enterprise-grade recommendations..."):
        try:
            # Generate recommendations
            results = calculator.generate_all_recommendations()
            df = pd.DataFrame.from_dict(results, orient='index')
            
            # Display summary metrics
            st.subheader(":trophy: Recommendation Summary")
            prod = results["PROD"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Instance Type</div><div class="metric-value">{prod["instance_type"]}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-title">vCPUs / RAM</div><div class="metric-value">{prod["vCPUs"]} / {prod["RAM_GB"]}GB</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><div class="metric-title">Monthly Cost</div><div class="metric-value">${prod["total_cost"]:,.2f}</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><div class="metric-title">TCO Savings</div><div class="metric-value">{prod["tco_savings"]:,.1f}%</div></div>', unsafe_allow_html=True)
            
            # Detailed recommendations
            st.subheader(":clipboard: Environment-Specific Recommendations")
            st.dataframe(
                df[["instance_type", "vCPUs", "RAM_GB", "storage_GB", "total_cost"]],
                column_config={
                    "total_cost": st.column_config.NumberColumn(
                        "Monthly Cost",
                        format="$%.2f"
                    )
                },
                use_container_width=True
            )
            
            # Advisories
            st.subheader(":warning: Optimization Advisories")
            for env in results:
                if "error" not in results[env] and results[env]["advisories"]:
                    st.markdown(f"**{env} Environment**")
                    for advisory in results[env]["advisories"]:
                        st.markdown(f'<div class="advisory-card">{advisory}</div>', unsafe_allow_html=True)
            
            # TCO Analysis
            st.subheader(":chart_with_downwards_trend: {}-Year TCO Comparison".format(years))
            tco_df = pd.DataFrame(calculator.tco_data)
            fig = px.line(
                tco_df, 
                x="Year", 
                y=["OnPrem", "Cloud"],
                labels={"value": "Cost ($)", "variable": "Deployment"},
                markers=True
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Year",
                yaxis_title="Cumulative Cost ($)",
                legend_title="Deployment Type"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk Assessment
            st.subheader(":triangular_flag_on_post: Risk Assessment Matrix")
            st.markdown("""
            <div class="risk-matrix">
                <div class="risk-cell" style="background-color:#10B981;grid-column:1;grid-row:1">Low Impact<br>Low Likelihood</div>
                <div class="risk-cell" style="background-color:#A7F3D0;grid-column:2;grid-row:1">Medium Impact<br>Low Likelihood</div>
                <div class="risk-cell" style="background-color:#FDE68A;grid-column:3;grid-row:1">High Impact<br>Low Likelihood</div>
                <div class="risk-cell" style="background-color:#FCD34D;grid-column:4;grid-row:1">Very High Impact<br>Low Likelihood</div>
                <div class="risk-cell" style="background-color:#FCA5A5;grid-column:5;grid-row:1">Critical Impact<br>Low Likelihood</div>
                
                <div class="risk-cell" style="background-color:#A7F3D0;grid-column:1;grid-row:2">Low Impact<br>Medium Likelihood</div>
                <div class="risk-cell" style="background-color:#FDE68A;grid-column:2;grid-row:2">Medium Impact<br>Medium Likelihood</div>
                <div class="risk-cell" style="background-color:#FCD34D;grid-column:3;grid-row:2">High Impact<br>Medium Likelihood</div>
                <div class="risk-cell" style="background-color:#FCA5A5;grid-column:4;grid-row:2">Very High Impact<br>Medium Likelihood</div>
                <div class="risk-cell" style="background-color:#EF4444;grid-column:5;grid-row:2">Critical Impact<br>Medium Likelihood</div>
                
                <div class="risk-cell" style="background-color:#FDE68A;grid-column:1;grid-row:3">Low Impact<br>High Likelihood</div>
                <div class="risk-cell" style="background-color:#FCD34D;grid-column:2;grid-row:3">Medium Impact<br>High Likelihood</div>
                <div class="risk-cell" style="background-color:#FCA5A5;grid-column:3;grid-row:3">High Impact<br>High Likelihood</div>
                <div class="risk-cell" style="background-color:#EF4444;grid-column:4;grid-row:3">Very High Impact<br>High Likelihood</div>
                <div class="risk-cell" style="background-color:#B91C1C;grid-column:5;grid-row:3">Critical Impact<br>High Likelihood</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Resource Forecast
            st.subheader(":crystal_ball: Resource Utilization Forecast")
            forecast_data = []
            current_storage = storage
            current_cores = cores
            current_iops = iops
            
            for year in range(1, years + 1):
                current_storage = current_storage * (1 + (growth/100))
                current_cores = current_cores * 1.15  # 15% annual growth
                current_iops = current_iops * 1.2  # 20% annual growth
                
                forecast_data.append({
                    "Year": year,
                    "Storage (GB)": current_storage,
                    "vCPUs": current_cores,
                    "IOPS": current_iops
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            fig = px.bar(
                forecast_df, 
                x="Year", 
                y=["Storage (GB)", "vCPUs", "IOPS"],
                barmode="group",
                labels={"value": "Resource Requirement", "variable": "Resource Type"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"✅ Recommendations generated in {time.time()-start_time:.2f} seconds")
            
        except Exception as e:
            st.error(f"🚨 Error generating recommendations: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**Enterprise-Grade Features**  
✔ Real-time AWS pricing  ✔ TCO analysis  ✔ Risk assessment matrix  ✔ Multi-engine support  
✔ Performance optimization  ✔ Compliance frameworks  ✔ HA/DR configurations  ✔ Multi-year projections  
""")