"""
Complete utility functions for AI Database Migration Studio
"""
import pandas as pd
import json
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

def parse_uploaded_file(uploaded_file):
    """Parse uploaded CSV/Excel file into a list of input dictionaries"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, ["Unsupported file format. Please upload CSV or Excel file."]
        
        print(f"=== DEBUG: Original DataFrame ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:\n{df.head()}")
        
    except Exception as e:
        return None, [f"Error reading file: {str(e)}"]
    
    # Required columns mapping
    required_columns = {
        'database_engine': 'engine',
        'aws_region': 'region',
        'cpu_cores': 'cores',
        'cpu_utilization': 'cpu_util',
        'ram_gb': 'ram',
        'ram_utilization': 'ram_util',
        'storage_gb': 'storage',
        'iops': 'iops'
    }
    
    # Optional columns with defaults
    optional_columns = {
    'growth_rate': ('growth', 15),
    'backup_days': ('backup_days', 7),
    'projection_years': ('years', 3),
    'data_transfer_gb': ('data_transfer_gb', 100),
    # Add Multi-AZ optional columns
    'multi_az_enabled': ('multi_az_enabled', False),
    'read_replica_count': ('read_replica_count', 2),
    'read_write_ratio': ('read_write_ratio', 70)
    }
    
    # Check for required columns
    missing_columns = [col for col in required_columns.keys() if col not in df.columns]
    if missing_columns:
        return None, [f"Missing required columns: {', '.join(missing_columns)}"]
    
    print(f"DEBUG: All required columns found: {list(required_columns.keys())}")
    
    # Create a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Rename columns to match input structure
    df_processed.rename(columns={k: v for k, v in required_columns.items()}, inplace=True)
    
    # Add optional columns with defaults
    for col, (new_name, default) in optional_columns.items():
        if col in df.columns:
            df_processed[new_name] = df[col]
        else:
            df_processed[new_name] = default
    
    # Add database name if exists
    if 'database_name' in df.columns:
        df_processed['db_name'] = df['database_name']
    else:
        df_processed['db_name'] = [f"Database {i+1}" for i in range(len(df_processed))]
    
    print(f"DEBUG: Processed DataFrame columns: {list(df_processed.columns)}")
    print(f"DEBUG: Sample processed row: {df_processed.iloc[0].to_dict()}")
    
    # Convert to list of dictionaries
    inputs_list = df_processed.to_dict(orient='records')
    
    print(f"DEBUG: Created {len(inputs_list)} input dictionaries")
    
    # Validate each input set
    valid_inputs = []
    errors = []
    
    for idx, input_data in enumerate(inputs_list):
        input_errors = validate_inputs(input_data)
        if not input_errors:
            valid_inputs.append(input_data)
            print(f"DEBUG: Row {idx+1} ({input_data.get('db_name', 'Unknown')}) - VALID")
        else:
            db_name = input_data.get('db_name', f"Row {idx+1}")
            error_msg = f"{db_name}: {', '.join(input_errors)}"
            errors.append(error_msg)
            print(f"DEBUG: Row {idx+1} ({db_name}) - INVALID: {input_errors}")
    
    print(f"DEBUG: Final result - {len(valid_inputs)} valid, {len(errors)} errors")
    
    return valid_inputs, errors

def export_full_report(all_results):
    """Export complete analysis report to Excel"""
    output = io.BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Create summary sheet
            summary_data = []
            for result in all_results:
                prod_rec = result['recommendations']['PROD']
                summary_data.append({
                    "Database": result['inputs'].get('db_name', 'N/A'),
                    "Engine": result['inputs'].get('engine', 'N/A'),
                    "Instance Type": prod_rec['instance_type'],
                    "vCPUs": prod_rec['vcpus'],
                    "RAM (GB)": prod_rec['ram_gb'],
                    "Storage (GB)": prod_rec['storage_gb'],
                    "Monthly Cost": prod_rec['monthly_cost'],
                    "Annual Cost": prod_rec['annual_cost'],
                    "Optimization Score": f"{prod_rec.get('optimization_score', 85)}%"
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create detailed sheets for each database
            for i, result in enumerate(all_results):
                db_name = result['inputs'].get('db_name', f'Database_{i+1}')
                # Clean sheet name for Excel (remove invalid characters and limit length)
                sheet_name = ''.join(c for c in db_name if c.isalnum() or c in ' -_')[:31]
                
                # Prepare detailed data
                detail_rows = []
                
                # Input parameters
                detail_rows.append(["CONFIGURATION", ""])
                for key, value in result['inputs'].items():
                    detail_rows.append([key.replace('_', ' ').title(), value])
                
                detail_rows.append(["", ""])
                detail_rows.append(["RECOMMENDATIONS", ""])
                
                # Environment recommendations
                for env, rec in result['recommendations'].items():
                    detail_rows.append([f"{env} Environment", ""])
                    detail_rows.append(["Instance Type", rec['instance_type']])
                    detail_rows.append(["vCPUs", rec['vcpus']])
                    detail_rows.append(["RAM (GB)", rec['ram_gb']])
                    detail_rows.append(["Storage (GB)", rec['storage_gb']])
                    detail_rows.append(["Monthly Cost", f"${rec['monthly_cost']:.2f}"])
                    detail_rows.append(["Annual Cost", f"${rec['annual_cost']:.2f}"])
                    detail_rows.append(["", ""])
                
                # AI insights if available
                if result.get('ai_insights'):
                    detail_rows.append(["AI INSIGHTS", ""])
                    ai_insights = result['ai_insights']
                    
                    if 'workload' in ai_insights and 'error' not in ai_insights['workload']:
                        workload = ai_insights['workload']
                        detail_rows.append(["Workload Type", workload.get('workload_type', 'N/A')])
                        detail_rows.append(["Complexity", workload.get('complexity', 'N/A')])
                        detail_rows.append(["Timeline", workload.get('timeline', 'N/A')])
                        
                        if workload.get('recommendations'):
                            detail_rows.append(["AI Recommendations", ""])
                            for rec in workload['recommendations'][:5]:
                                detail_rows.append(["", rec])
                
                # Convert to DataFrame and save
                detail_df = pd.DataFrame(detail_rows, columns=['Parameter', 'Value'])
                detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        output.seek(0)
        return output
        
    except Exception as e:
        print(f"Error creating Excel report: {str(e)}")
        # Return a simple CSV as fallback
        summary_data = []
        for result in all_results:
            prod_rec = result['recommendations']['PROD']
            summary_data.append({
                "Database": result['inputs'].get('db_name', 'N/A'),
                "Engine": result['inputs'].get('engine', 'N/A'),
                "Instance_Type": prod_rec['instance_type'],
                "Monthly_Cost": prod_rec['monthly_cost']
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_output = io.StringIO()
        summary_df.to_csv(csv_output, index=False)
        return io.BytesIO(csv_output.getvalue().encode())

def validate_inputs(inputs: Dict) -> List[str]:
    """Validate user inputs and return list of errors"""
    errors = []
    
    # Required fields validation
    required_fields = {
        'cores': 'CPU Cores',
        'ram': 'RAM (GB)',
        'storage': 'Storage (GB)',
        'cpu_util': 'CPU Utilization',
        'ram_util': 'RAM Utilization'
    }
    
    for field, display_name in required_fields.items():
        if field not in inputs:
            errors.append(f"{display_name} is required")
        elif pd.isna(inputs[field]) or inputs[field] is None:
            errors.append(f"{display_name} cannot be empty")
        elif not isinstance(inputs[field], (int, float)) or inputs[field] <= 0:
            errors.append(f"{display_name} must be a positive number (got: {inputs[field]})")
    
    # Range validations (only if values exist and are numeric)
    if 'cpu_util' in inputs and isinstance(inputs['cpu_util'], (int, float)):
        if inputs['cpu_util'] > 100:
            errors.append("CPU utilization cannot exceed 100%")
    
    if 'ram_util' in inputs and isinstance(inputs['ram_util'], (int, float)):
        if inputs['ram_util'] > 100:
            errors.append("RAM utilization cannot exceed 100%")
    
    if 'growth' in inputs and isinstance(inputs['growth'], (int, float)):
        if inputs['growth'] < 0:
            errors.append("Growth rate cannot be negative")
        elif inputs['growth'] > 1000:
            errors.append("Growth rate seems unrealistic (>1000%)")
    
    # Logical validations (only if values exist and are numeric)
    if 'cores' in inputs and isinstance(inputs['cores'], (int, float)):
        if inputs['cores'] > 1000:
            errors.append("CPU cores count seems unrealistic (>1000)")
    
    if 'ram' in inputs and isinstance(inputs['ram'], (int, float)):
        if inputs['ram'] > 10000:
            errors.append("RAM amount seems unrealistic (>10TB)")
    
    if 'storage' in inputs and isinstance(inputs['storage'], (int, float)):
        if inputs['storage'] > 1000000:
            errors.append("Storage amount seems unrealistic (>1PB)")
    
    # Engine validation
    valid_engines = ['oracle-ee', 'oracle-se', 'postgres', 'aurora-postgresql', 'aurora-mysql', 'sqlserver']
    if 'engine' in inputs and inputs['engine'] not in valid_engines:
        errors.append(f"Unsupported database engine: {inputs['engine']}. Valid options: {', '.join(valid_engines)}")
    
    # Region validation
    valid_regions = ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
    if 'region' in inputs and inputs['region'] not in valid_regions:
        errors.append(f"Unsupported AWS region: {inputs['region']}. Valid options: {', '.join(valid_regions)}")
    
    return errors

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amounts with proper formatting"""
    if currency == "USD":
        return f"${amount:,.2f}"
    return f"{amount:,.2f} {currency}"

def format_large_number(num: float, suffix: str = "") -> str:
    """Format large numbers with K, M, B suffixes"""
    if num >= 1e9:
        return f"{num/1e9:.1f}B{suffix}"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M{suffix}"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K{suffix}"
    else:
        return f"{num:.1f}{suffix}"

def calculate_roi(annual_savings: float, annual_cost: float) -> float:
    """Calculate Return on Investment percentage"""
    if annual_cost == 0:
        return 0
    return (annual_savings / annual_cost) * 100

def calculate_payback_period(total_investment: float, monthly_savings: float) -> float:
    """Calculate payback period in months"""
    if monthly_savings <= 0:
        return float('inf')
    return total_investment / monthly_savings

def generate_recommendation_score(
    cpu_efficiency: float, 
    ram_efficiency: float, 
    cost_efficiency: float,
    performance_score: float = 1.0
) -> int:
    """Generate recommendation score (0-100) based on multiple factors"""
    weights = {
        'cpu': 0.3,
        'ram': 0.25, 
        'cost': 0.3,
        'performance': 0.15
    }
    
    score = (
        cpu_efficiency * weights['cpu'] + 
        ram_efficiency * weights['ram'] + 
        cost_efficiency * weights['cost'] +
        performance_score * weights['performance']
    ) * 100
    
    return min(100, max(0, int(score)))

def get_instance_recommendations(vcpus: int, ram: int, engine: str) -> List[Dict[str, str]]:
    """Get instance type recommendations based on requirements"""
    recommendations = []
    
    # T3 instances for light workloads
    if vcpus <= 2 and ram <= 8:
        recommendations.append({
            "type": "db.t3.medium",
            "reason": "Cost-effective for variable workloads with CPU credits",
            "use_case": "Development and testing environments"
        })
    
    # M5 instances for balanced workloads
    if vcpus <= 8 and ram <= 32:
        recommendations.append({
            "type": "db.m5.xlarge", 
            "reason": "Balanced compute and memory for general workloads",
            "use_case": "Production applications with steady performance needs"
        })
    
    # R5 instances for memory-intensive workloads
    if ram >= 16:
        recommendations.append({
            "type": "db.r5.large",
            "reason": "Memory-optimized for demanding applications",
            "use_case": "Analytics and memory-intensive databases"
        })
    
    # Aurora Serverless for variable workloads
    if engine.startswith('aurora'):
        recommendations.append({
            "type": "Aurora Serverless",
            "reason": "Automatic scaling for variable workloads",
            "use_case": "Applications with intermittent or unpredictable usage"
        })
    
    # High-performance instances for large workloads
    if vcpus >= 16 or ram >= 64:
        recommendations.append({
            "type": "db.r5.4xlarge",
            "reason": "High-performance for enterprise workloads",
            "use_case": "Large-scale production databases"
        })
    
    return recommendations

def calculate_storage_costs(
    storage_gb: int, 
    storage_type: str = "gp3", 
    iops: int = 3000,
    throughput_mbps: int = 125
) -> Dict[str, float]:
    """Calculate detailed storage costs"""
    costs = {}
    
    if storage_type == "gp3":
        # GP3 base cost
        costs["base_storage"] = storage_gb * 0.08
        
        # Additional IOPS cost (free up to 3000)
        extra_iops = max(0, iops - 3000)
        costs["additional_iops"] = extra_iops * 0.005
        
        # Additional throughput cost (free up to 125 MB/s)
        extra_throughput = max(0, throughput_mbps - 125)
        costs["additional_throughput"] = extra_throughput * 0.04
        
    elif storage_type == "gp2":
        costs["base_storage"] = storage_gb * 0.10
        costs["additional_iops"] = 0  # Burstable
        costs["additional_throughput"] = 0
        
    elif storage_type == "io1":
        costs["base_storage"] = storage_gb * 0.125
        costs["additional_iops"] = iops * 0.065
        costs["additional_throughput"] = 0
        
    elif storage_type == "io2":
        costs["base_storage"] = storage_gb * 0.125
        costs["additional_iops"] = iops * 0.065
        costs["additional_throughput"] = 0
    
    costs["total"] = sum(costs.values())
    return costs

def export_to_json(data: Dict, filename: str = None) -> str:
    """Export data to JSON string or file"""
    json_str = json.dumps(data, indent=2, default=str)
    
    if filename:
        with open(filename, 'w') as f:
            f.write(json_str)
        return f"Data exported to {filename}"
    else:
        return json_str

def export_to_csv(data: List[Dict], filename: str = None) -> str:
    """Export data to CSV string or file"""
    df = pd.DataFrame(data)
    
    if filename:
        df.to_csv(filename, index=False)
        return f"Data exported to {filename}"
    else:
        return df.to_csv(index=False)

def create_download_link(data: str, filename: str, mime_type: str = "text/plain") -> str:
    """Create a download link for data"""
    b64_data = base64.b64encode(data.encode()).decode()
    return f'<a href="data:{mime_type};base64,{b64_data}" download="{filename}">Download {filename}</a>'

def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} EB"

def calculate_network_transfer_time(
    data_size_gb: int, 
    bandwidth_mbps: int = 1000,
    efficiency_factor: float = 0.8
) -> Dict[str, float]:
    """Calculate estimated data transfer time"""
    
    # Convert GB to Mb (Gigabytes to Megabits)
    data_size_mb = data_size_gb * 8 * 1024
    
    # Calculate transfer time with efficiency factor
    theoretical_time_hours = data_size_mb / (bandwidth_mbps * 60)
    actual_time_hours = theoretical_time_hours / efficiency_factor
    
    return {
        "theoretical_hours": theoretical_time_hours,
        "estimated_hours": actual_time_hours,
        "estimated_days": actual_time_hours / 24,
        "bandwidth_mbps": bandwidth_mbps,
        "efficiency_factor": efficiency_factor
    }

def validate_api_key(api_key: str) -> bool:
    """Validate Claude API key format"""
    if not api_key:
        return False
    
    # Claude API keys start with 'sk-ant-'
    if not api_key.startswith('sk-ant-'):
        return False
    
    # Basic length check
    if len(api_key) < 20:
        return False
    
    return True

def get_optimization_recommendations(
    current_config: Dict,
    recommended_config: Dict,
    cost_analysis: Dict
) -> List[Dict[str, str]]:
    """Generate optimization recommendations based on analysis"""
    
    recommendations = []
    
    # Instance optimization
    if recommended_config.get('vcpus', 0) < current_config.get('cores', 0):
        savings = (current_config['cores'] - recommended_config['vcpus']) * 50  # Rough estimate
        recommendations.append({
            "category": "Compute Optimization",
            "recommendation": f"Right-size from {current_config['cores']} to {recommended_config['vcpus']} vCPUs",
            "impact": f"Save ~${savings}/month",
            "effort": "Low"
        })
    
    # Storage optimization
    if cost_analysis.get('storage_type') == 'gp2':
        recommendations.append({
            "category": "Storage Optimization", 
            "recommendation": "Migrate to GP3 storage for better price/performance",
            "impact": "20% storage cost reduction",
            "effort": "Low"
        })
    
    # Reserved Instance recommendation
    recommendations.append({
        "category": "Cost Optimization",
        "recommendation": "Consider 1-year Reserved Instances for stable workloads",
        "impact": "30-40% cost reduction",
        "effort": "Low"
    })
    
    # Auto Scaling recommendation
    recommendations.append({
        "category": "Performance Optimization",
        "recommendation": "Implement auto-scaling for read replicas",
        "impact": "Better performance during peak loads",
        "effort": "Medium"
    })
    
    return recommendations