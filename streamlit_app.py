import streamlit as st
import pandas as pd
import time
import traceback
import json
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from io import BytesIO
import base64

class RealTimeRDSSizingCalculator:
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
        self.aws_available = self._check_aws_credentials()
        self.pricing_client = None
        self.pricing_cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 3600  # 1 hour cache
        self.recommendations = {}
        self.inputs = {}
        self.bulk_results = {}
        
        if self.aws_available:
            try:
                self.pricing_client = boto3.client('pricing', region_name='us-east-1')
                st.success("✅ AWS credentials detected - real-time pricing enabled")
            except Exception as e:
                self.aws_available = False
                st.warning(f"⚠️ AWS client initialization failed: {e}")
    
    def _check_aws_credentials(self):
        """Check if AWS credentials are available"""
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            return credentials is not None
        except Exception:
            return False
    
    def _fetch_real_time_pricing(self, engine, region):
        """Fetch real-time pricing from AWS Pricing API"""
        cache_key = f"{engine}_{region}"
        
        # Check cache first
        if (cache_key in self.pricing_cache and 
            cache_key in self.cache_timestamp and
            time.time() - self.cache_timestamp[cache_key] < self.cache_duration):
            return self.pricing_cache[cache_key]
        
        try:
            # Map engine names to AWS filter values
            engine_mapping = {
                'postgres': 'PostgreSQL',
                'aurora-postgresql': 'Aurora PostgreSQL',
                'aurora-mysql': 'Aurora MySQL',
                'oracle-ee': 'Oracle',
                'oracle-se': 'Oracle',
                'sqlserver': 'SQL Server'
            }
            
            aws_engine = engine_mapping.get(engine, 'PostgreSQL')
            
            # Build filters for AWS Pricing API
            filters = [
                {'Type': 'TERM_MATCH', 'Field': 'regionCode', 'Value': region},
                {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': aws_engine},
                {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': 'Single-AZ'},
            ]
            
            # Add license model filter
            if 'oracle' not in engine and 'sqlserver' not in engine:
                filters.append({'Type': 'TERM_MATCH', 'Field': 'licenseModel', 'Value': 'No License required'})
            
            instances = []
            next_token = None
            max_pages = 3  # Limit to prevent long waits
            page_count = 0
            
            while page_count < max_pages:
                try:
                    if next_token:
                        response = self.pricing_client.get_products(
                            ServiceCode='AmazonRDS',
                            Filters=filters,
                            NextToken=next_token,
                            MaxResults=50
                        )
                    else:
                        response = self.pricing_client.get_products(
                            ServiceCode='AmazonRDS',
                            Filters=filters,
                            MaxResults=50
                        )
                    
                    for price_item in response['PriceList']:
                        try:
                            product = json.loads(price_item)
                            attrs = product['product']['attributes']
                            
                            instance_type = attrs.get('instanceType')
                            if not instance_type or 'db.' not in instance_type:
                                continue
                            
                            # Get pricing information
                            terms = product.get('terms', {}).get('OnDemand', {})
                            if not terms:
                                continue
                            
                            price_dim = next(iter(terms.values())).get('priceDimensions', {})
                            if not price_dim:
                                continue
                            
                            price_info = next(iter(price_dim.values()))
                            price_usd = price_info.get('pricePerUnit', {}).get('USD', '0')
                            
                            if float(price_usd) == 0:
                                continue
                            
                            # Extract instance specifications
                            vcpu_str = attrs.get('vcpu', '0')
                            memory_str = attrs.get('memory', '0 GiB')
                            
                            # Parse vCPU
                            try:
                                vcpu = int(vcpu_str) if vcpu_str.isdigit() else int(float(vcpu_str))
                            except:
                                continue
                            
                            # Parse memory
                            try:
                                memory = float(memory_str.split()[0])
                            except:
                                continue
                            
                            instance_data = {
                                "type": instance_type,
                                "vCPU": vcpu,
                                "memory": memory,
                                "pricing": {"ondemand": float(price_usd)},
                                "source": "AWS_API"
                            }
                            
                            instances.append(instance_data)
                            
                        except Exception as e:
                            continue  # Skip problematic entries
                    
                    next_token = response.get('NextToken')
                    if not next_token:
                        break
                    
                    page_count += 1
                    
                except ClientError as e:
                    break
            
            if instances:
                # Sort by instance type for consistency
                instances.sort(key=lambda x: x['type'])
                
                # Cache the results
                self.pricing_cache[cache_key] = instances
                self.cache_timestamp[cache_key] = time.time()
                
                return instances
            else:
                return self._get_fallback_pricing(engine, region)
                
        except Exception as e:
            return self._get_fallback_pricing(engine, region)
    
    def _get_fallback_pricing(self, engine, region):
        """Get fallback pricing data when AWS API is not available"""
        
        # Enhanced fallback data
        fallback_data = {
            "postgres": [
                {"type": "db.t3.micro", "vCPU": 2, "memory": 1, "pricing": {"ondemand": 0.026}, "source": "fallback"},
                {"type": "db.t3.small", "vCPU": 2, "memory": 2, "pricing": {"ondemand": 0.051}, "source": "fallback"},
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.102}, "source": "fallback"},
                {"type": "db.t3.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.204}, "source": "fallback"},
                {"type": "db.t3.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand": 0.408}, "source": "fallback"},
                {"type": "db.t3.2xlarge", "vCPU": 8, "memory": 32, "pricing": {"ondemand": 0.816}, "source": "fallback"},
                {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.192}, "source": "fallback"},
                {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand": 0.384}, "source": "fallback"},
                {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "pricing": {"ondemand": 0.768}, "source": "fallback"},
                {"type": "db.m5.4xlarge", "vCPU": 16, "memory": 64, "pricing": {"ondemand": 1.536}, "source": "fallback"},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.24}, "source": "fallback"},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 0.48}, "source": "fallback"},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand": 0.96}, "source": "fallback"},
                {"type": "db.r5.4xlarge", "vCPU": 16, "memory": 128, "pricing": {"ondemand": 1.92}, "source": "fallback"},
            ],
            "aurora-postgresql": [
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.082}, "source": "fallback"},
                {"type": "db.t4g.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.073}, "source": "fallback"},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.285}, "source": "fallback"},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 0.57}, "source": "fallback"},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand": 1.14}, "source": "fallback"},
                {"type": "db.r5.4xlarge", "vCPU": 16, "memory": 128, "pricing": {"ondemand": 2.28}, "source": "fallback"},
                {"type": "db.r6g.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.256}, "source": "fallback"},
                {"type": "db.r6g.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 0.512}, "source": "fallback"},
                {"type": "db.r6g.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand": 1.024}, "source": "fallback"},
            ],
            "oracle-ee": [
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "pricing": {"ondemand": 0.272}, "source": "fallback"},
                {"type": "db.t3.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.544}, "source": "fallback"},
                {"type": "db.m5.large", "vCPU": 2, "memory": 8, "pricing": {"ondemand": 0.475}, "source": "fallback"},
                {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "pricing": {"ondemand": 0.95}, "source": "fallback"},
                {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "pricing": {"ondemand": 1.90}, "source": "fallback"},
                {"type": "db.m5.4xlarge", "vCPU": 16, "memory": 64, "pricing": {"ondemand": 3.80}, "source": "fallback"},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "pricing": {"ondemand": 0.60}, "source": "fallback"},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "pricing": {"ondemand": 1.20}, "source": "fallback"},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "pricing": {"ondemand": 2.40}, "source": "fallback"},
            ]
        }
        
        # Default to postgres if engine not found
        engine_data = fallback_data.get(engine, fallback_data["postgres"])
        
        # Adjust pricing for different regions
        region_multipliers = {
            "us-east-1": 1.0,
            "us-west-1": 1.08,
            "us-west-2": 1.08,
            "eu-west-1": 1.15,
            "ap-southeast-1": 1.20,
            "ap-northeast-1": 1.18,
            "eu-central-1": 1.12
        }
        
        multiplier = region_multipliers.get(region, 1.1)
        
        # Apply region multiplier
        adjusted_data = []
        for instance in engine_data:
            adjusted_instance = instance.copy()
            adjusted_instance["pricing"] = {
                "ondemand": instance["pricing"]["ondemand"] * multiplier
            }
            adjusted_data.append(adjusted_instance)
        
        return adjusted_data
    
    def _get_instance_data(self, engine, region):
        """Get instance data with real-time pricing when available"""
        if self.aws_available and self.pricing_client:
            return self._fetch_real_time_pricing(engine, region)
        else:
            return self._get_fallback_pricing(engine, region)
    
    def _parse_read_write_ratio(self, workload_pattern=None, read_write_ratio=None):
        """Parse read/write ratio from inputs"""
        if workload_pattern and workload_pattern in self.WORKLOAD_PATTERNS:
            pattern = self.WORKLOAD_PATTERNS[workload_pattern]
            return pattern["read_percentage"], pattern["write_percentage"]
        
        # Parse custom ratio like "60:40"
        ratio_str = read_write_ratio or "60:40"
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
    
    def calculate_requirements(self, env, workload_inputs):
        """Calculate requirements for a specific environment with reader/writer logic"""
        profile = self.ENV_PROFILES[env]
        deployment_config = self.DEPLOYMENT_OPTIONS[workload_inputs["deployment"]]
        
        # Calculate base requirements
        base_vcpus = workload_inputs["on_prem_cores"] * (workload_inputs["peak_cpu_percent"] / 100)
        base_ram = workload_inputs["on_prem_ram_gb"] * (workload_inputs["peak_ram_percent"] / 100)
        
        # Apply environment-specific factors
        env_vcpus = base_vcpus * profile["cpu_factor"] * profile["performance_headroom"]
        env_ram = base_ram * profile["ram_factor"] * profile["performance_headroom"]
        
        # Calculate read/write workload distribution
        read_pct, write_pct = self._parse_read_write_ratio(
            workload_inputs.get("workload_pattern"), 
            workload_inputs.get("read_write_ratio")
        )
        
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
        
        # Get available instances (with real-time pricing)
        instances = self._get_instance_data(workload_inputs["engine"], workload_inputs["region"])
        
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
                "total_reader_cost": reader_instance["pricing"]["ondemand"] * 24 * 30 * deployment_config["reader_count"],
                "pricing_source": reader_instance.get("source", "unknown")
            }
        
        # Calculate storage and costs
        storage = self._calculate_storage(env, workload_inputs)
        costs = self._calculate_costs_with_readers(writer_instance, reader_recommendations, storage, env)
        
        # Generate advisories
        advisories = self._generate_advisories(writer_instance, reader_recommendations, env, read_pct, write_pct, workload_inputs)
        
        return {
            "environment": env,
            "deployment_type": workload_inputs["deployment"],
            "workload_pattern": f"{read_pct:.0f}% reads, {write_pct:.0f}% writes",
            
            # Writer information
            "writer": {
                "instance_type": writer_instance["type"],
                "vCPUs": final_writer_cpu,
                "RAM_GB": final_writer_ram,
                "actual_vCPUs": writer_instance["vCPU"],
                "actual_RAM_GB": writer_instance["memory"],
                "monthly_cost": writer_instance["pricing"]["ondemand"] * 24 * 30,
                "pricing_source": writer_instance.get("source", "unknown")
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
            "has_readers": deployment_config["has_readers"],
            "pricing_source": writer_instance.get("source", "unknown")
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
    
    def _calculate_storage(self, env, workload_inputs):
        """Calculate storage requirements"""
        profile = self.ENV_PROFILES[env]
        base_storage = workload_inputs["storage_current_gb"]
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
    
    def _generate_advisories(self, writer_instance, reader_recommendations, env, read_pct, write_pct, workload_inputs):
        """Generate optimization advisories"""
        advisories = []
        
        # Pricing source advisory
        pricing_source = writer_instance.get("source", "unknown")
        if pricing_source == "AWS_API":
            advisories.append("✅ Using real-time AWS pricing data")
        elif pricing_source == "fallback":
            advisories.append("⚠️ Using fallback pricing - configure AWS credentials for real-time pricing")
        
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
            if workload_inputs.get("deployment") == "Single-AZ":
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
        if "aurora" in workload_inputs["engine"] and reader_recommendations:
            advisories.append("🚀 Consider Aurora Auto Scaling for readers based on CPU utilization")
        
        return advisories
    
    def generate_all_recommendations(self, workload_inputs=None):
        """Generate recommendations for all environments"""
        if workload_inputs is None:
            workload_inputs = self.inputs
            
        self.recommendations = {}
        
        for env in self.ENV_PROFILES:
            try:
                self.recommendations[env] = self.calculate_requirements(env, workload_inputs)
            except Exception as e:
                self.recommendations[env] = {"error": str(e)}
        
        return self.recommendations
    
    def process_bulk_workloads(self, workloads_df):
        """Process multiple workloads from a DataFrame"""
        self.bulk_results = {}
        
        for index, row in workloads_df.iterrows():
            workload_name = row.get('workload_name', f'Workload_{index + 1}')
            
            try:
                # Map CSV columns to internal format
                workload_inputs = {
                    "region": row.get('region', 'us-east-1'),
                    "engine": row.get('engine', 'postgres'),
                    "deployment": row.get('deployment', 'Multi-AZ'),
                    "workload_pattern": row.get('workload_pattern', 'OLTP_BALANCED'),
                    "read_write_ratio": row.get('read_write_ratio', '60:40'),
                    "on_prem_cores": int(row.get('on_prem_cores', 8)),
                    "peak_cpu_percent": float(row.get('peak_cpu_percent', 70)),
                    "on_prem_ram_gb": int(row.get('on_prem_ram_gb', 32)),
                    "peak_ram_percent": float(row.get('peak_ram_percent', 80)),
                    "storage_current_gb": int(row.get('storage_current_gb', 250)),
                    "storage_growth_rate": float(row.get('storage_growth_rate', 20)) / 100
                }
                
                # Generate recommendations for this workload
                workload_recommendations = {}
                for env in self.ENV_PROFILES:
                    try:
                        workload_recommendations[env] = self.calculate_requirements(env, workload_inputs)
                    except Exception as e:
                        workload_recommendations[env] = {"error": str(e)}
                
                self.bulk_results[workload_name] = {
                    "inputs": workload_inputs,
                    "recommendations": workload_recommendations
                }
                
            except Exception as e:
                self.bulk_results[workload_name] = {
                    "inputs": {},
                    "recommendations": {"error": f"Processing error: {str(e)}"}
                }
        
        return self.bulk_results

def get_bulk_template():
    """Generate a template CSV for bulk upload"""
    template_data = {
        'workload_name': ['Web Application', 'Data Warehouse', 'Development DB'],
        'region': ['us-east-1', 'us-west-2', 'eu-west-1'],
        'engine': ['postgres', 'aurora-postgresql', 'postgres'],
        'deployment': ['Multi-AZ', 'Aurora Global', 'Single-AZ'],
        'workload_pattern': ['OLTP_BALANCED', 'READ_HEAVY', 'MIXED'],
        'read_write_ratio': ['60:40', '85:15', '50:50'],
        'on_prem_cores': [16, 32, 4],
        'peak_cpu_percent': [70, 85, 50],
        'on_prem_ram_gb': [64, 128, 16],
        'peak_ram_percent': [75, 80, 60],
        'storage_current_gb': [500, 2000, 100],
        'storage_growth_rate': [15, 25, 10]
    }
    
    return pd.DataFrame(template_data)

def create_bulk_results_summary(bulk_results):
    """Create a summary DataFrame from bulk results"""
    summary_data = []
    
    for workload_name, workload_data in bulk_results.items():
        recommendations = workload_data.get('recommendations', {})
        inputs = workload_data.get('inputs', {})
        
        for env, result in recommendations.items():
            if 'error' not in result:
                row = {
                    'Workload': workload_name,
                    'Environment': env,
                    'Region': inputs.get('region', 'N/A'),
                    'Engine': inputs.get('engine', 'N/A'),
                    'Deployment': inputs.get('deployment', 'N/A'),
                    'Writer Instance': result['writer']['instance_type'],
                    'Writer Cost': result['instance_cost'],
                    'Reader Instance': result['readers']['instance_type'] if result.get('readers') else 'None',
                    'Reader Count': result['readers']['count'] if result.get('readers') else 0,
                    'Reader Cost': result.get('reader_cost', 0),
                    'Storage (GB)': result['storage_GB'],
                    'Storage Cost': result['storage_cost'],
                    'Total Cost': result['total_cost'],
                    'Pricing Source': result.get('pricing_source', 'unknown')
                }
                summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def create_workload_charts(inputs):
    """Create workload visualization charts"""
    
    # Read/Write Workload Distribution Pie Chart
    read_pct, write_pct = 60, 40
    if inputs.get("workload_pattern") and inputs["workload_pattern"] in calculator.WORKLOAD_PATTERNS:
        pattern = calculator.WORKLOAD_PATTERNS[inputs["workload_pattern"]]
        read_pct = pattern["read_percentage"]
        write_pct = pattern["write_percentage"]
    
    fig_workload = go.Figure(data=[go.Pie(
        labels=['Read Operations', 'Write Operations'],
        values=[read_pct, write_pct],
        hole=.3,
        marker_colors=['#36A2EB', '#FF6384'],
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig_workload.update_layout(
        title={
            'text': f"📊 Workload Distribution - {inputs.get('workload_pattern', 'BALANCED').replace('_', ' ').title()}",
            'x': 0.5,
            'font': {'size': 16}
        },
        showlegend=True,
        font=dict(size=12),
        height=350
    )
    
    return fig_workload

def create_utilization_gauges(inputs):
    """Create CPU and RAM utilization gauge charts"""
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=["CPU Utilization", "RAM Utilization"]
    )
    
    # CPU Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=inputs.get("peak_cpu_percent", 70),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Peak CPU %"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 50], 'color': "#d4edda"},
                {'range': [50, 75], 'color': "#fff3cd"},
                {'range': [75, 100], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ), row=1, col=1)
    
    # RAM Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=inputs.get("peak_ram_percent", 80),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Peak RAM %"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#ff7f0e"},
            'steps': [
                {'range': [0, 60], 'color': "#d4edda"},
                {'range': [60, 80], 'color': "#fff3cd"},
                {'range': [80, 100], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ), row=1, col=2)
    
    fig.update_layout(
        title={
            'text': "⚡ Current Resource Utilization",
            'x': 0.5,
            'font': {'size': 16}
        },
        height=350,
        font=dict(size=12)
    )
    
    return fig

def create_storage_projection_chart(inputs):
    """Create storage growth projection chart"""
    
    current_storage = inputs.get("storage_current_gb", 250)
    growth_rate = inputs.get("storage_growth_rate", 0.2)
    
    # Project 5 years into the future
    months = list(range(0, 61, 6))  # Every 6 months for 5 years
    storage_values = [current_storage * ((1 + growth_rate) ** (month / 12)) for month in months]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=months,
        y=storage_values,
        mode='lines+markers',
        name='Projected Storage',
        line=dict(color='#2E8B57', width=3),
        marker=dict(size=8),
        fill='tonexty',
        fillcolor='rgba(46, 139, 87, 0.1)'
    ))
    
    # Add current point
    fig.add_trace(go.Scatter(
        x=[0],
        y=[current_storage],
        mode='markers',
        name='Current Storage',
        marker=dict(size=12, color='red', symbol='diamond')
    ))
    
    fig.update_layout(
        title={
            'text': f"💾 Storage Growth Projection ({growth_rate*100:.0f}% annual growth)",
            'x': 0.5,
            'font': {'size': 16}
        },
        xaxis_title="Months from Now",
        yaxis_title="Storage (GB)",
        showlegend=True,
        height=350,
        font=dict(size=12)
    )
    
    return fig

def create_cost_breakdown_pie(result):
    """Create cost breakdown pie chart for a single environment"""
    
    labels = ['Writer Instance', 'Storage', 'Backup']
    values = [result['instance_cost'], result['storage_cost'], result['storage_cost'] * 0.25]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    if result.get('reader_cost', 0) > 0:
        labels.insert(1, 'Reader Instances')
        values.insert(1, result['reader_cost'])
        colors.insert(1, '#96CEB4')
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker_colors=colors,
        textinfo='label+percent+value',
        texttemplate='%{label}<br>%{percent}<br>$%{value:,.0f}',
        textfont_size=10
    )])
    
    fig.update_layout(
        title={
            'text': f"💰 Cost Breakdown - {result['environment']} Environment",
            'x': 0.5,
            'font': {'size': 14}
        },
        showlegend=True,
        height=300,
        font=dict(size=10)
    )
    
    return fig

def create_environment_comparison_radar(results):
    """Create radar chart comparing environments"""
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if len(valid_results) < 2:
        return None
    
    categories = ['Cost Score', 'Performance Score', 'Storage Score', 'Reliability Score']
    
    fig = go.Figure()
    
    # Normalize scores (0-100 scale)
    max_cost = max(result['total_cost'] for result in valid_results.values())
    
    for env, result in valid_results.items():
        # Calculate normalized scores
        cost_score = 100 - (result['total_cost'] / max_cost * 100)  # Lower cost = higher score
        performance_score = min(100, (result['writer']['actual_vCPUs'] / result['writer']['vCPUs']) * 100)
        storage_score = min(100, (result['storage_GB'] / 1000) * 100)  # Arbitrary scaling
        reliability_score = 90 if result.get('readers') else 70  # Multi-AZ gets higher score
        
        scores = [cost_score, performance_score, storage_score, reliability_score]
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name=env,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        title={
            'text': "🎯 Environment Comparison Radar",
            'x': 0.5,
            'font': {'size': 16}
        },
        showlegend=True,
        height=400
    )
    
    return fig

def create_instance_comparison_chart(results):
    """Create instance comparison chart"""
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    data = []
    for env, result in valid_results.items():
        data.append({
            'Environment': env,
            'Instance Type': result['writer']['instance_type'],
            'vCPUs': result['writer']['actual_vCPUs'],
            'RAM (GB)': result['writer']['actual_RAM_GB'],
            'Monthly Cost': result['instance_cost'],
            'Has Readers': 'Yes' if result.get('readers') else 'No',
            'Reader Count': result['readers']['count'] if result.get('readers') else 0
        })
    
    df = pd.DataFrame(data)
    
    # Create scatter plot: vCPUs vs RAM, sized by cost, colored by environment
    fig = px.scatter(
        df, 
        x='vCPUs', 
        y='RAM (GB)',
        size='Monthly Cost',
        color='Environment',
        hover_data=['Instance Type', 'Monthly Cost', 'Has Readers'],
        title="💻 Instance Specifications Comparison",
        size_max=30
    )
    
    fig.update_layout(
        height=400,
        font=dict(size=12),
        title={'x': 0.5, 'font': {'size': 16}}
    )
    
    return fig

def create_bulk_cost_heatmap(bulk_results):
    """Create cost heatmap for bulk results"""
    
    # Prepare data for heatmap
    workloads = []
    environments = ['DEV', 'QA', 'SQA', 'PROD']
    cost_matrix = []
    
    for workload_name, workload_data in bulk_results.items():
        recommendations = workload_data.get('recommendations', {})
        workloads.append(workload_name)
        
        row = []
        for env in environments:
            if env in recommendations and 'error' not in recommendations[env]:
                cost = recommendations[env]['total_cost']
                row.append(cost)
            else:
                row.append(0)
        cost_matrix.append(row)
    
    if not cost_matrix:
        return None
    
    fig = go.Figure(data=go.Heatmap(
        z=cost_matrix,
        x=environments,
        y=workloads,
        colorscale='RdYlBu_r',
        text=[[f'${cost:,.0f}' if cost > 0 else 'N/A' for cost in row] for row in cost_matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Monthly Cost ($)")
    ))
    
    fig.update_layout(
        title={
            'text': "🔥 Cost Heatmap - All Workloads vs Environments",
            'x': 0.5,
            'font': {'size': 16}
        },
        xaxis_title="Environment",
        yaxis_title="Workload",
        height=max(300, len(workloads) * 50),
        font=dict(size=12)
    )
    
    return fig

def create_deployment_comparison_pie():
    """Create deployment options comparison"""
    
    deployment_data = []
    for deployment, config in calculator.DEPLOYMENT_OPTIONS.items():
        deployment_data.append({
            'Deployment': deployment,
            'Cost Multiplier': config['cost_multiplier'],
            'Reader Count': config['reader_count'],
            'Description': config['description']
        })
    
    df = pd.DataFrame(deployment_data)
    
    fig = px.pie(
        df,
        values='Cost Multiplier',
        names='Deployment',
        title="🏗️ Deployment Options - Relative Cost Impact",
        hover_data=['Reader Count', 'Description']
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        height=350,
        font=dict(size=12),
        title={'x': 0.5, 'font': {'size': 16}}
    )
    
    return fig

# Configure Streamlit
st.set_page_config(
    page_title="AWS RDS Sizing Tool with Advanced Analytics",
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
    .pricing-indicator {
        background: #e8f5e8;
        border: 1px solid #4caf50;
        border-radius: 4px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .setup-info {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .bulk-info {
        background: #e7f3ff;
        border: 1px solid #2196f3;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .chart-container {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize calculator
@st.cache_resource
def get_calculator():
    return RealTimeRDSSizingCalculator()

calculator = get_calculator()

# Header
st.title("🚀 AWS RDS & Aurora Sizing Tool")
st.markdown("**Enhanced with Real-time Pricing, Advanced Analytics & Interactive Visualizations**")

# Pricing Status Indicator
if calculator.aws_available:
    st.markdown("""
    <div class="pricing-indicator">
        ✅ <strong>Real-time AWS Pricing:</strong> ENABLED - Prices are fetched directly from AWS Pricing API
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="setup-info">
        ⚠️ <strong>Real-time AWS Pricing:</strong> DISABLED - Using fallback pricing. Configure AWS credentials for real-time pricing.
    </div>
    """, unsafe_allow_html=True)

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔧 Single Workload Sizing", "📂 Bulk Upload Sizing", "📊 Analytics Dashboard"])

with tab1:
    # Single workload sizing (existing functionality)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # AWS Settings
        with st.expander("☁️ AWS Settings", expanded=True):
            region = st.selectbox("Region", [
                "us-east-1", "us-west-1", "us-west-2", 
                "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"
            ], index=0)
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

    # Pre-calculation visualizations
    st.header("📊 Workload Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        workload_chart = create_workload_charts(calculator.inputs)
        st.plotly_chart(workload_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        utilization_chart = create_utilization_gauges(calculator.inputs)
        st.plotly_chart(utilization_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        storage_chart = create_storage_projection_chart(calculator.inputs)
        st.plotly_chart(storage_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Main Content
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🚀 Generate Sizing with Real-time Pricing", type="primary", use_container_width=True):
            with st.spinner("🔄 Calculating sizing and fetching pricing..."):
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
                    
                    # Show pricing source info
                    if results:
                        sample_result = next(iter(results.values()))
                        if 'error' not in sample_result:
                            pricing_source = sample_result.get('pricing_source', 'unknown')
                            if pricing_source == 'AWS_API':
                                st.success("💡 Using real-time AWS pricing data")
                            else:
                                st.info("📝 Using fallback pricing data")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

    with col2:
        if st.button("📊 Export CSV", use_container_width=True):
            if 'results' in st.session_state:
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
                            'Total Cost': result['total_cost'],
                            'Pricing Source': result.get('pricing_source', 'unknown')
                        }
                        export_data.append(row)
                    
                    df_export = pd.DataFrame(export_data)
                    csv = df_export.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"rds_single_sizing_{int(time.time())}.csv",
                        mime="text/csv"
                    )

    with col3:
        if st.button("🔄 Refresh Pricing", use_container_width=True):
            if calculator.aws_available:
                calculator.pricing_cache.clear()
                calculator.cache_timestamp.clear()
                st.success("✅ Pricing cache cleared - next calculation will fetch fresh data")
            else:
                st.warning("⚠️ AWS credentials not configured")

    # Display Results with Enhanced Visualizations
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
                pricing_source = prod_result.get('pricing_source', 'unknown')
                source_emoji = "🟢" if pricing_source == "AWS_API" else "📝"
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{source_emoji}</div>
                    <div class="metric-label">{pricing_source.replace('_', ' ').title()}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced Visualizations Section
            st.header("📈 Advanced Analytics & Visualizations")
            
            # Row 1: Environment comparison and instance analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                radar_chart = create_environment_comparison_radar(valid_results)
                if radar_chart:
                    st.plotly_chart(radar_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                instance_chart = create_instance_comparison_chart(valid_results)
                st.plotly_chart(instance_chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Row 2: Cost breakdown pies for each environment
            st.subheader("💰 Cost Breakdown by Environment")
            
            env_cols = st.columns(len(valid_results))
            for idx, (env, result) in enumerate(valid_results.items()):
                with env_cols[idx]:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    cost_pie = create_cost_breakdown_pie(result)
                    st.plotly_chart(cost_pie, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced Results Display with Reader/Writer Breakdown
            st.header("📋 Environment-Specific Recommendations")
            
            for env, result in valid_results.items():
                with st.expander(f"{env} Environment - ${result['total_cost']:,.2f}/month", expanded=True):
                    
                    # Pricing source indicator
                    pricing_source = result.get('pricing_source', 'unknown')
                    if pricing_source == 'AWS_API':
                        st.success("✅ Real-time AWS pricing")
                    else:
                        st.info("📝 Fallback pricing")
                    
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
                            <p><strong>Pricing Source:</strong> {writer.get('pricing_source', 'unknown').replace('_', ' ').title()}</p>
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
                                <p><strong>Pricing Source:</strong> {readers.get('pricing_source', 'unknown').replace('_', ' ').title()}</p>
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
                            if "real-time AWS pricing" in advisory:
                                st.success(advisory)
                            elif "fallback pricing" in advisory:
                                st.warning(advisory)
                            else:
                                st.info(advisory)
            
            # Enhanced Charts Section
            st.header("📈 Enhanced Cost Analysis")
            
            # Create Plotly charts
            chart_data = []
            for env, result in valid_results.items():
                chart_data.append({
                    'Environment': env,
                    'Writer Cost': result['instance_cost'],
                    'Reader Cost': result.get('reader_cost', 0),
                    'Storage Cost': result['storage_cost'],
                    'Total Cost': result['total_cost'],
                    'Pricing Source': result.get('pricing_source', 'unknown')
                })
            
            df = pd.DataFrame(chart_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
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
                    title='💸 Cost Breakdown by Environment',
                    barmode='stack',
                    xaxis_title='Environment',
                    yaxis_title='Monthly Cost ($)',
                    font=dict(size=12),
                    title_x=0.5
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Total cost comparison with pricing source indicators
                colors = ['green' if source == 'AWS_API' else 'orange' for source in df['Pricing Source']]
                
                fig2 = go.Figure(data=[
                    go.Bar(
                        x=df['Environment'],
                        y=df['Total Cost'],
                        marker_color=colors,
                        text=[f"${cost:,.0f}" for cost in df['Total Cost']],
                        textposition='auto'
                    )
                ])
                
                fig2.update_layout(
                    title='💰 Total Monthly Cost by Environment<br><sub>Green: Real-time | Orange: Fallback</sub>',
                    xaxis_title='Environment',
                    yaxis_title='Monthly Cost ($)',
                    font=dict(size=12),
                    title_x=0.5
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Error Summary
            error_results = {k: v for k, v in results.items() if 'error' in v}
            if error_results:
                st.header("❌ Errors")
                for env, result in error_results.items():
                    st.error(f"{env}: {result['error']}")

with tab2:
    # Bulk upload sizing (existing functionality but enhanced with more charts)
    st.header("📂 Bulk Workload Sizing")
    
    st.markdown("""
    <div class="bulk-info">
        <h4>🚀 Bulk Upload Feature</h4>
        <p>Upload a CSV or Excel file containing multiple workloads to get sizing recommendations for all at once.</p>
        <p><strong>Benefits:</strong></p>
        <ul>
            <li>✅ Process multiple workloads simultaneously</li>
            <li>✅ Compare costs across different workloads and environments</li>
            <li>✅ Export comprehensive reports with advanced visualizations</li>
            <li>✅ Use real-time AWS pricing for all workloads</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Template download section
    st.subheader("📄 Download Template")
    col1, col2 = st.columns(2)
    
    with col1:
        template_df = get_bulk_template()
        
        csv_template = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv_template,
            file_name="rds_bulk_sizing_template.csv",
            mime="text/csv",
            help="Download a CSV template with sample workloads"
        )
    
    with col2:
        # Excel template
        buffer = BytesIO()
        template_df.to_excel(buffer, index=False, engine='openpyxl')
        buffer.seek(0)
        
        st.download_button(
            label="📥 Download Excel Template",
            data=buffer.getvalue(),
            file_name="rds_bulk_sizing_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download an Excel template with sample workloads"
        )
    
    # Show template preview
    st.subheader("👀 Template Preview")
    st.dataframe(template_df, use_container_width=True)
    
    # File upload section
    st.subheader("📤 Upload Your Workloads")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a file with your workload configurations. Use the template above for the correct format."
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} workload(s)")
            
            # Show uploaded data preview
            st.subheader("📊 Uploaded Data Preview")
            st.dataframe(df, use_container_width=True)
            
            # Validate required columns
            required_columns = ['workload_name', 'region', 'engine', 'deployment', 
                              'on_prem_cores', 'peak_cpu_percent', 'on_prem_ram_gb', 
                              'peak_ram_percent', 'storage_current_gb']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.info("Please ensure your file includes all required columns. Download the template for reference.")
            else:
                # Process button
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if st.button("🚀 Process All Workloads", type="primary", use_container_width=True):
                        with st.spinner(f"🔄 Processing {len(df)} workload(s) with real-time pricing..."):
                            start_time = time.time()
                            
                            try:
                                bulk_results = calculator.process_bulk_workloads(df)
                                st.session_state['bulk_results'] = bulk_results
                                st.session_state['bulk_generation_time'] = time.time() - start_time
                                
                                # Count successful vs failed workloads
                                successful = sum(1 for workload_data in bulk_results.values() 
                                               if not any('error' in rec for rec in workload_data.get('recommendations', {}).values()))
                                
                                st.success(f"✅ Processed {successful}/{len(df)} workloads successfully in {st.session_state['bulk_generation_time']:.1f} seconds")
                                
                                # Show pricing source info
                                if bulk_results:
                                    sample_workload = next(iter(bulk_results.values()))
                                    sample_rec = next(iter(sample_workload.get('recommendations', {}).values()))
                                    if 'error' not in sample_rec:
                                        pricing_source = sample_rec.get('pricing_source', 'unknown')
                                        if pricing_source == 'AWS_API':
                                            st.success("💡 Using real-time AWS pricing data for all workloads")
                                        else:
                                            st.info("📝 Using fallback pricing data for all workloads")
                                
                            except Exception as e:
                                st.error(f"❌ Error processing workloads: {str(e)}")
                                with st.expander("Error Details"):
                                    st.code(traceback.format_exc())
                
                with col2:
                    # Export bulk results
                    if 'bulk_results' in st.session_state:
                        bulk_results = st.session_state['bulk_results']
                        summary_df = create_bulk_results_summary(bulk_results)
                        
                        if not summary_df.empty:
                            csv_bulk = summary_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Export Results CSV",
                                data=csv_bulk,
                                file_name=f"rds_bulk_results_{int(time.time())}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                
                with col3:
                    if st.button("🔄 Clear Cache", use_container_width=True):
                        if calculator.aws_available:
                            calculator.pricing_cache.clear()
                            calculator.cache_timestamp.clear()
                            st.success("✅ Cache cleared")
                        else:
                            st.warning("⚠️ AWS not configured")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    # Display bulk results with enhanced visualizations
    if 'bulk_results' in st.session_state:
        bulk_results = st.session_state['bulk_results']
        
        st.header("📊 Bulk Processing Results")
        
        # Summary metrics
        total_workloads = len(bulk_results)
        successful_workloads = sum(1 for workload_data in bulk_results.values() 
                                  if not any('error' in rec for rec in workload_data.get('recommendations', {}).values()))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Workloads", total_workloads)
        with col2:
            st.metric("Successful", successful_workloads)
        with col3:
            st.metric("Failed", total_workloads - successful_workloads)
        with col4:
            processing_time = st.session_state.get('bulk_generation_time', 0)
            st.metric("Processing Time", f"{processing_time:.1f}s")
        
        # Enhanced visualizations for bulk results
        st.header("📈 Advanced Bulk Analytics")
        
        # Cost Heatmap
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        heatmap_chart = create_bulk_cost_heatmap(bulk_results)
        if heatmap_chart:
            st.plotly_chart(heatmap_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create summary table
        summary_df = create_bulk_results_summary(bulk_results)
        
        if not summary_df.empty:
            st.subheader("📋 Results Summary")
            st.dataframe(summary_df, use_container_width=True)
            
            # Enhanced cost analysis charts for bulk results
            st.subheader("📈 Enhanced Cost Analysis Across Workloads")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Group by environment for comparison
                fig_bulk = px.bar(
                    summary_df, 
                    x='Workload', 
                    y='Total Cost', 
                    color='Environment',
                    title='💰 Total Monthly Costs by Workload and Environment',
                    text='Total Cost'
                )
                fig_bulk.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                fig_bulk.update_layout(
                    height=500,
                    font=dict(size=12),
                    title_x=0.5
                )
                st.plotly_chart(fig_bulk, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                # Workload distribution by engine
                engine_dist = summary_df.groupby('Engine')['Total Cost'].mean().reset_index()
                fig_engine = px.pie(
                    engine_dist,
                    values='Total Cost',
                    names='Engine',
                    title='🔧 Average Cost Distribution by Engine Type'
                )
                fig_engine.update_layout(
                    height=500,
                    font=dict(size=12),
                    title_x=0.5
                )
                st.plotly_chart(fig_engine, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Cost breakdown by workload (Production only)
            prod_data = summary_df[summary_df['Environment'] == 'PROD']
            if not prod_data.empty:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig_prod = go.Figure()
                
                fig_prod.add_trace(go.Bar(
                    name='Writer Cost',
                    x=prod_data['Workload'],
                    y=prod_data['Writer Cost'],
                    marker_color='#2196F3'
                ))
                
                fig_prod.add_trace(go.Bar(
                    name='Reader Cost',
                    x=prod_data['Workload'],
                    y=prod_data['Reader Cost'],
                    marker_color='#9C27B0'
                ))
                
                fig_prod.add_trace(go.Bar(
                    name='Storage Cost',
                    x=prod_data['Workload'],
                    y=prod_data['Storage Cost'],
                    marker_color='#4CAF50'
                ))
                
                fig_prod.update_layout(
                    title='🏭 Production Environment - Cost Breakdown by Workload',
                    barmode='stack',
                    xaxis_title='Workload',
                    yaxis_title='Monthly Cost ($)',
                    height=400,
                    font=dict(size=12),
                    title_x=0.5
                )
                
                st.plotly_chart(fig_prod, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Regional cost comparison
            region_data = summary_df.groupby(['Region', 'Environment'])['Total Cost'].mean().reset_index()
            if len(region_data['Region'].unique()) > 1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig_region = px.bar(
                    region_data,
                    x='Region',
                    y='Total Cost',
                    color='Environment',
                    title='🌍 Average Costs by Region and Environment',
                    text='Total Cost'
                )
                fig_region.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                fig_region.update_layout(
                    height=400,
                    font=dict(size=12),
                    title_x=0.5
                )
                st.plotly_chart(fig_region, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed workload results
            st.subheader("🔍 Detailed Results by Workload")
            
            for workload_name, workload_data in bulk_results.items():
                recommendations = workload_data.get('recommendations', {})
                inputs = workload_data.get('inputs', {})
                
                # Count successful environments for this workload
                successful_envs = sum(1 for result in recommendations.values() if 'error' not in result)
                total_envs = len(recommendations)
                
                with st.expander(f"🖥️ {workload_name} ({successful_envs}/{total_envs} environments processed)", expanded=False):
                    
                    # Show workload configuration
                    st.subheader("⚙️ Configuration")
                    config_col1, config_col2 = st.columns(2)
                    
                    with config_col1:
                        st.write(f"**Engine:** {inputs.get('engine', 'N/A')}")
                        st.write(f"**Region:** {inputs.get('region', 'N/A')}")
                        st.write(f"**Deployment:** {inputs.get('deployment', 'N/A')}")
                        st.write(f"**Workload Pattern:** {inputs.get('workload_pattern', 'N/A')}")
                    
                    with config_col2:
                        st.write(f"**CPU Cores:** {inputs.get('on_prem_cores', 'N/A')}")
                        st.write(f"**Peak CPU:** {inputs.get('peak_cpu_percent', 'N/A')}%")
                        st.write(f"**RAM:** {inputs.get('on_prem_ram_gb', 'N/A')}GB")
                        st.write(f"**Storage:** {inputs.get('storage_current_gb', 'N/A')}GB")
                    
                    # Show results for each environment
                    st.subheader("📊 Environment Results")
                    
                    for env, result in recommendations.items():
                        if 'error' not in result:
                            env_col1, env_col2, env_col3, env_col4 = st.columns(4)
                            
                            with env_col1:
                                st.metric(f"{env} Writer", result['writer']['instance_type'])
                            with env_col2:
                                if result.get('readers'):
                                    reader_info = f"{result['readers']['instance_type']} x{result['readers']['count']}"
                                else:
                                    reader_info = "None"
                                st.metric(f"{env} Readers", reader_info)
                            with env_col3:
                                st.metric(f"{env} Total Cost", f"${result['total_cost']:,.2f}")
                            with env_col4:
                                pricing_source = result.get('pricing_source', 'unknown')
                                source_emoji = "🟢" if pricing_source == "AWS_API" else "📝"
                                st.metric("Pricing", f"{source_emoji} {pricing_source}")
                        else:
                            st.error(f"❌ {env}: {result['error']}")
        
        else:
            st.warning("⚠️ No successful results to display")

with tab3:
    # New Analytics Dashboard
    st.header("📊 Analytics Dashboard")
    
    st.markdown("""
    <div class="chart-container">
        <h4>📈 Global Analytics & Insights</h4>
        <p>Comprehensive analytics dashboard providing insights into deployment options, pricing trends, and optimization opportunities.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Deployment options overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        deployment_chart = create_deployment_comparison_pie()
        st.plotly_chart(deployment_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Workload patterns distribution
        pattern_data = []
        for pattern, info in calculator.WORKLOAD_PATTERNS.items():
            pattern_data.append({
                'Pattern': pattern.replace('_', ' ').title(),
                'Read %': info['read_percentage'],
                'Write %': info['write_percentage']
            })
        
        pattern_df = pd.DataFrame(pattern_data)
        
        fig_patterns = go.Figure()
        
        fig_patterns.add_trace(go.Bar(
            name='Read %',
            x=pattern_df['Pattern'],
            y=pattern_df['Read %'],
            marker_color='#36A2EB'
        ))
        
        fig_patterns.add_trace(go.Bar(
            name='Write %',
            x=pattern_df['Pattern'],
            y=pattern_df['Write %'],
            marker_color='#FF6384'
        ))
        
        fig_patterns.update_layout(
            title='📊 Workload Pattern Characteristics',
            barmode='stack',
            xaxis_title='Pattern Type',
            yaxis_title='Percentage (%)',
            font=dict(size=12),
            title_x=0.5,
            height=350
        )
        
        st.plotly_chart(fig_patterns, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Environment factors comparison
    st.subheader("🏗️ Environment Scaling Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Environment scaling factors
        env_data = []
        for env, profile in calculator.ENV_PROFILES.items():
            env_data.append({
                'Environment': env,
                'CPU Factor': profile['cpu_factor'],
                'RAM Factor': profile['ram_factor'],
                'Storage Factor': profile['storage_factor'],
                'Performance Headroom': profile['performance_headroom']
            })
        
        env_df = pd.DataFrame(env_data)
        
        fig_env = go.Figure()
        
        for factor in ['CPU Factor', 'RAM Factor', 'Storage Factor', 'Performance Headroom']:
            fig_env.add_trace(go.Scatter(
                x=env_df['Environment'],
                y=env_df[factor],
                mode='lines+markers',
                name=factor,
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig_env.update_layout(
            title='⚖️ Environment Scaling Factors',
            xaxis_title='Environment',
            yaxis_title='Factor Value',
            font=dict(size=12),
            title_x=0.5,
            height=350
        )
        
        st.plotly_chart(fig_env, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Cost efficiency analysis
        efficiency_data = []
        for env, profile in calculator.ENV_PROFILES.items():
            # Calculate efficiency score (inverse of factors - lower resource usage = higher efficiency)
            efficiency_score = (2 - profile['cpu_factor']) * (2 - profile['ram_factor']) * (2 - profile['storage_factor'])
            efficiency_data.append({
                'Environment': env,
                'Efficiency Score': efficiency_score,
                'Resource Usage': profile['cpu_factor'] + profile['ram_factor'] + profile['storage_factor']
            })
        
        efficiency_df = pd.DataFrame(efficiency_data)
        
        fig_efficiency = px.scatter(
            efficiency_df,
            x='Resource Usage',
            y='Efficiency Score',
            size='Efficiency Score',
            color='Environment',
            title='⚡ Environment Efficiency Analysis',
            hover_data=['Environment'],
            size_max=30
        )
        
        fig_efficiency.update_layout(
            font=dict(size=12),
            title_x=0.5,
            height=350
        )
        
        st.plotly_chart(fig_efficiency, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Regional cost comparison (simulated)
    st.subheader("🌍 Regional Cost Analysis")
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Create sample regional cost data
    regions = ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"]
    instance_types = ["db.t3.medium", "db.m5.large", "db.r5.large", "db.r5.xlarge"]
    
    regional_data = []
    for region in regions:
        for instance in instance_types:
            # Simulate base cost with regional multipliers
            base_cost = {"db.t3.medium": 102, "db.m5.large": 192, "db.r5.large": 240, "db.r5.xlarge": 480}[instance]
            multipliers = {
                "us-east-1": 1.0, "us-west-1": 1.08, "us-west-2": 1.08,
                "eu-west-1": 1.15, "eu-central-1": 1.12,
                "ap-southeast-1": 1.20, "ap-northeast-1": 1.18
            }
            cost = base_cost * multipliers[region]
            
            regional_data.append({
                'Region': region,
                'Instance Type': instance,
                'Monthly Cost': cost
            })
    
    regional_df = pd.DataFrame(regional_data)
    
    fig_regional = px.bar(
        regional_df,
        x='Region',
        y='Monthly Cost',
        color='Instance Type',
        title='💸 Regional Cost Comparison by Instance Type',
        text='Monthly Cost'
    )
    
    fig_regional.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
    fig_regional.update_layout(
        height=500,
        font=dict(size=12),
        title_x=0.5,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_regional, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Engine comparison analysis
    st.subheader("🔧 Database Engine Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Engine licensing comparison
        engine_info = [
            {'Engine': 'PostgreSQL', 'License Cost': 'Free', 'Relative Cost': 1.0, 'Use Case': 'General Purpose'},
            {'Engine': 'Aurora PostgreSQL', 'License Cost': 'Free', 'Relative Cost': 1.2, 'Use Case': 'Cloud Native'},
            {'Engine': 'Aurora MySQL', 'License Cost': 'Free', 'Relative Cost': 1.1, 'Use Case': 'Web Apps'},
            {'Engine': 'Oracle EE', 'License Cost': 'High', 'Relative Cost': 3.5, 'Use Case': 'Enterprise'},
            {'Engine': 'Oracle SE', 'License Cost': 'Medium', 'Relative Cost': 2.0, 'Use Case': 'Business'},
            {'Engine': 'SQL Server', 'License Cost': 'Medium', 'Relative Cost': 2.2, 'Use Case': 'Microsoft Stack'}
        ]
        
        engine_df = pd.DataFrame(engine_info)
        
        fig_engines = px.pie(
            engine_df,
            values='Relative Cost',
            names='Engine',
            title='🔧 Engine Cost Distribution',
            hover_data=['License Cost', 'Use Case']
        )
        
        fig_engines.update_traces(textposition='inside', textinfo='percent+label')
        fig_engines.update_layout(
            height=400,
            font=dict(size=10),
            title_x=0.5
        )
        
        st.plotly_chart(fig_engines, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Performance vs Cost analysis
        performance_data = [
            {'Engine': 'PostgreSQL', 'Performance Score': 85, 'Cost Score': 95, 'Popularity': 90},
            {'Engine': 'Aurora PostgreSQL', 'Performance Score': 95, 'Cost Score': 80, 'Popularity': 85},
            {'Engine': 'Aurora MySQL', 'Performance Score': 90, 'Cost Score': 85, 'Popularity': 80},
            {'Engine': 'Oracle EE', 'Performance Score': 100, 'Cost Score': 40, 'Popularity': 60},
            {'Engine': 'Oracle SE', 'Performance Score': 90, 'Cost Score': 60, 'Popularity': 50},
            {'Engine': 'SQL Server', 'Performance Score': 88, 'Cost Score': 55, 'Popularity': 70}
        ]
        
        perf_df = pd.DataFrame(performance_data)
        
        fig_perf = px.scatter(
            perf_df,
            x='Cost Score',
            y='Performance Score',
            size='Popularity',
            color='Engine',
            title='⚡ Engine Performance vs Cost Analysis',
            size_max=25,
            hover_data=['Engine', 'Popularity']
        )
        
        fig_perf.update_layout(
            height=400,
            font=dict(size=12),
            title_x=0.5
        )
        
        st.plotly_chart(fig_perf, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Storage growth projection analysis
    st.subheader("💾 Storage Growth Impact Analysis")
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    # Create storage growth scenarios
    growth_rates = [10, 15, 20, 25, 30, 40, 50]
    initial_storage = 250
    years = 5
    
    growth_data = []
    for rate in growth_rates:
        final_storage = initial_storage * ((1 + rate/100) ** years)
        storage_cost_monthly = final_storage * 0.10
        annual_storage_cost = storage_cost_monthly * 12
        
        growth_data.append({
            'Growth Rate (%)': rate,
            'Final Storage (GB)': final_storage,
            'Monthly Storage Cost': storage_cost_monthly,
            'Annual Storage Cost': annual_storage_cost
        })
    
    growth_df = pd.DataFrame(growth_data)
    
    fig_growth = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Storage Size Growth", "Cost Impact"],
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Storage growth
    fig_growth.add_trace(
        go.Scatter(
            x=growth_df['Growth Rate (%)'],
            y=growth_df['Final Storage (GB)'],
            mode='lines+markers',
            name='Storage Size',
            line=dict(color='#2E8B57', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Cost impact
    fig_growth.add_trace(
        go.Scatter(
            x=growth_df['Growth Rate (%)'],
            y=growth_df['Annual Storage Cost'],
            mode='lines+markers',
            name='Annual Cost',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    fig_growth.update_layout(
        title_text="📈 5-Year Storage Growth Impact Analysis",
        height=400,
        font=dict(size=12),
        title_x=0.5
    )
    
    fig_growth.update_xaxes(title_text="Annual Growth Rate (%)", row=1, col=1)
    fig_growth.update_xaxes(title_text="Annual Growth Rate (%)", row=1, col=2)
    fig_growth.update_yaxes(title_text="Storage Size (GB)", row=1, col=1)
    fig_growth.update_yaxes(title_text="Annual Cost ($)", row=1, col=2)
    
    st.plotly_chart(fig_growth, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Cost optimization recommendations
    st.subheader("💡 Cost Optimization Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="chart-container">
            <h4>💰 Cost Optimization Tips</h4>
            <ul>
                <li><strong>Environment Sizing:</strong> Use DEV/QA factors to reduce non-production costs by 50-75%</li>
                <li><strong>Engine Selection:</strong> PostgreSQL offers best cost-performance for most workloads</li>
                <li><strong>Read Replicas:</strong> Distribute read load to optimize writer instance sizing</li>
                <li><strong>Storage Growth:</strong> Monitor growth rates - 20%+ annual growth significantly impacts costs</li>
                <li><strong>Regional Choice:</strong> US-East-1 typically offers lowest pricing</li>
                <li><strong>Instance Right-sizing:</strong> Aurora auto-scaling can optimize reader costs</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="chart-container">
            <h4>🎯 Performance Optimization Tips</h4>
            <ul>
                <li><strong>Read/Write Distribution:</strong> 80%+ read workloads benefit most from read replicas</li>
                <li><strong>Memory Optimization:</strong> R5 instances for memory-intensive workloads</li>
                <li><strong>CPU Optimization:</strong> M5 instances for balanced compute needs</li>
                <li><strong>Multi-AZ Benefits:</strong> Provides HA with minimal performance impact</li>
                <li><strong>Aurora Global:</strong> Best for global applications with <100ms latency needs</li>
                <li><strong>Workload Patterns:</strong> Match instance types to workload characteristics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# AWS Credentials Setup Instructions
with st.expander("🔧 AWS Credentials Setup for Real-time Pricing"):
    st.markdown("""
    To enable real-time pricing, configure AWS credentials in Streamlit Cloud:
    
    **Option 1: Streamlit Secrets (Recommended for Cloud)**
    Add to your `.streamlit/secrets.toml`:
    ```toml
    [aws]
    access_key_id = "your_access_key"
    secret_access_key = "your_secret_key"
    region = "us-east-1"
    ```
    
    **Option 2: Environment Variables**
    Set in your deployment environment:
    - `AWS_ACCESS_KEY_ID`
    - `AWS_SECRET_ACCESS_KEY`
    - `AWS_DEFAULT_REGION`
    
    **Required IAM Permissions:**
    ```json
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "pricing:GetProducts",
                    "pricing:DescribeServices"
                ],
                "Resource": "*"
            }
        ]
    }
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
**🎯 Enhanced Features:**
- ✅ **Advanced Visualizations**: Interactive charts, pie charts, heatmaps, radar charts, and gauges
- ✅ **Single & Bulk Workload Processing**: Handle individual workloads or process multiple workloads from CSV/Excel files
- ✅ **Real-time AWS Pricing**: Live pricing from AWS Pricing API with regional variations
- ✅ **Reader/Writer Optimization**: Separate sizing for Multi-AZ deployments with intelligent distribution
- ✅ **Smart Analytics Dashboard**: Comprehensive insights into deployment options, regional costs, and optimization opportunities
- ✅ **Interactive Cost Analysis**: Pie charts, heatmaps, and comparative visualizations for better decision making
- ✅ **Workload Pattern Visualization**: Gauges, distribution charts, and storage growth projections
- ✅ **Environment Comparison**: Radar charts and scatter plots for multi-dimensional analysis
- ✅ **Smart Caching**: 1-hour cache for performance with manual refresh option
- ✅ **Fallback Support**: Graceful degradation when AWS API is unavailable
- ✅ **Comprehensive Export**: Export individual or bulk results with detailed cost breakdowns

**🔧 Bulk Upload Benefits:**
- 📊 Process dozens of workloads simultaneously with enhanced visualizations
- 💰 Compare costs across different workloads and environments with interactive charts
- 📈 Generate comprehensive reports with heatmaps, pie charts, and trend analysis
- ⚡ Use real-time AWS pricing for accurate cost estimates
- 🎯 Advanced analytics dashboard for strategic planning and optimization

**📊 Visualization Features:**
- 🥧 **Pie Charts**: Cost breakdowns, engine distributions, deployment comparisons
- 📊 **Bar Charts**: Environment comparisons, regional costs, workload analysis
- 🔥 **Heatmaps**: Multi-workload cost analysis across environments
- 📡 **Radar Charts**: Multi-dimensional environment comparisons
- ⚡ **Gauges**: Resource utilization and performance indicators
- 📈 **Line Charts**: Storage growth projections and trend analysis
- 🎯 **Scatter Plots**: Performance vs cost analysis and efficiency comparisons
- 📋 **Interactive Tables**: Sortable, filterable data with export capabilities
""")