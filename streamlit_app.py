import streamlit as st
import pandas as pd
import plotly.express as px
from rds_sizing import RDSDatabaseSizingCalculator
from report_generator import ReportGenerator
import time
import traceback

# Configure enterprise-grade UI
st.set_page_config(
    page_title="Enterprise RDS/Aurora Sizing",
    layout="wide",
    page_icon="📊"
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
        color: #1F2937 !important;
        font-weight: 500;
    }
    
    .risk-matrix {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        grid-template-rows: repeat(3, 1fr);
        gap: 8px;
        margin-top: 1rem;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .risk-cell {
        padding: 1rem;
        text-align: center;
        border-radius: 8px;
        font-size: 0.8rem;
        min-height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize calculator
try:
    calculator = RDSDatabaseSizingCalculator()
    report_generator = ReportGenerator()
except Exception as e:
    st.error(f"Error initializing calculator: {str(e)}")
    st.stop()

# App header
st.title("🚀 Enterprise AWS RDS & Aurora Sizing Tool")
st.markdown("""
**Comprehensive migration planning for Oracle, PostgreSQL, Aurora with TCO analysis and optimization recommendations**  
*Enterprise-grade solution with real-time AWS pricing and risk assessment*
""")

# Sidebar configuration
with st.sidebar:
    st.header("☁️ AWS Configuration")
    region = st.selectbox("AWS Region", ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "ap-southeast-1"], index=0)
    
    st.header("⚙️ Database Settings")
    engine = st.selectbox("Database Engine", calculator.ENGINES, index=0)
    
    # Deployment model selection for supported engines
    supported_serverless = ['aurora-postgresql', 'aurora-mysql']
    if engine in supported_serverless:
        deployment_model = st.radio("Deployment Model", ["Provisioned", "Serverless"])
    else:
        deployment_model = "Provisioned"
    
    deployment = st.selectbox("Deployment Type", list(calculator.DEPLOYMENT_OPTIONS.keys()), index=1)
    storage_type = st.selectbox("Storage Type", list(calculator.STORAGE_TYPES.keys()), index=1)
    
    st.header("📈 Workload Profile")
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
    
    st.header("💰 Financials")
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
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    generate_btn = st.button("🚀 Generate Sizing Recommendations", type="primary", use_container_width=True)

with col2:
    download_excel = st.button("📊 Download Excel", use_container_width=True)

with col3:
    download_pdf = st.button("📄 Download PDF", use_container_width=True)

if generate_btn:
    start_time = time.time()
    
    with st.spinner("🚀 Generating enterprise-grade recommendations..."):
        try:
            # Generate recommendations
            results = calculator.generate_all_recommendations()
            
            # Check if we have valid results
            valid_results = {k: v for k, v in results.items() if "error" not in v}
            if not valid_results:
                st.error("❌ No valid recommendations could be generated. Please check your input parameters.")
                for env, result in results.items():
                    if "error" in result:
                        st.error(f"Error in {env}: {result['error']}")
                st.stop()
            
            # Store results in session state for download buttons
            st.session_state['results'] = results
            st.session_state['calculator'] = calculator
            
            # Create DataFrame for display
            df_data = []
            for env, rec in valid_results.items():
                df_data.append({
                    'Environment': env,
                    'Instance Type': rec['instance_type'],
                    'vCPUs': rec['vCPUs'],
                    'RAM (GB)': rec['RAM_GB'],
                    'Storage (GB)': rec['storage_GB'],
                    'Monthly Cost': rec['total_cost']
                })
            df = pd.DataFrame(df_data)
            
            # Display summary metrics
            st.subheader("🏆 Recommendation Summary")
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
            st.subheader("📋 Environment-Specific Recommendations")
            st.dataframe(
                df.set_index('Environment'),
                column_config={
                    "Monthly Cost": st.column_config.NumberColumn(
                        "Monthly Cost",
                        format="$%.2f"
                    )
                },
                use_container_width=True
            )
            
            # Advisories
            st.subheader("⚠️ Optimization Advisories")
            advisory_found = False
            for env in results:
                if "error" not in results[env] and results[env]["advisories"]:
                    advisory_found = True
                    st.markdown(f"**{env} Environment**")
                    for advisory in results[env]["advisories"]:
                        st.markdown(f'<div class="advisory-card" style="color: #1F2937 !important;">{advisory}</div>', unsafe_allow_html=True)
            
            if not advisory_found:
                st.info("✅ No optimization advisories - your configuration looks good!")
            
            # TCO Analysis
            if hasattr(calculator, 'tco_data') and calculator.tco_data:
                st.subheader(f"📉 {years}-Year TCO Comparison")
                tco_df = pd.DataFrame(calculator.tco_data)
                fig = px.line(
                    tco_df, 
                    x="Year", 
                    y=["OnPrem", "Cloud"],
                    labels={"value": "Cost ($)", "variable": "Deployment"},
                    markers=True,
                    title="Total Cost of Ownership Comparison"
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Year",
                    yaxis_title="Cumulative Cost ($)",
                    legend_title="Deployment Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Resource Forecast
            st.subheader("🔮 Resource Utilization Forecast")
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
            fig2 = px.bar(
                forecast_df, 
                x="Year", 
                y=["Storage (GB)", "vCPUs", "IOPS"],
                barmode="group",
                labels={"value": "Resource Requirement", "variable": "Resource Type"},
                title="Projected Resource Growth"
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            st.success(f"✅ Recommendations generated in {time.time()-start_time:.2f} seconds")
            
        except Exception as e:
            st.error(f"🚨 Error generating recommendations: {str(e)}")
            st.error("Please check your input parameters and try again.")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

# Download handlers
if download_excel and 'calculator' in st.session_state:
    try:
        excel_data = report_generator.generate_excel_report(st.session_state['calculator'])
        st.download_button(
            label="📊 Download Excel Report",
            data=excel_data,
            file_name=f"aws_rds_sizing_report_{st.session_state['calculator'].inputs['engine']}_{st.session_state['calculator'].inputs['region']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Error generating Excel report: {str(e)}")

if download_pdf and 'calculator' in st.session_state:
    try:
        pdf_data = report_generator.generate_pdf_report(st.session_state['calculator'])
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_data,
            file_name=f"aws_rds_sizing_report_{st.session_state['calculator'].inputs['engine']}_{st.session_state['calculator'].inputs['region']}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**🌟 Enterprise-Grade Features**  
✔ Real-time AWS pricing  ✔ TCO analysis  ✔ Risk assessment matrix  ✔ Multi-engine support  
✔ Performance optimization  ✔ Compliance frameworks  ✔ HA/DR configurations  ✔ Multi-year projections  

**💡 Supported Database Engines:** Oracle EE/SE, PostgreSQL, Aurora PostgreSQL/MySQL, SQL Server  
**🌍 Supported Regions:** US East/West, EU West, AP Southeast  
**📊 Export Formats:** Excel, PDF, CSV reports with detailed analysis  
""")