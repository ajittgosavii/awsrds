import math
import json
import logging
from datetime import datetime
from functools import lru_cache

class RDSDatabaseSizingCalculator:
    ENGINES = ['oracle-ee', 'oracle-se', 'postgres', 'aurora-postgresql', 'aurora-mysql', 'sqlserver']
    
    # Enhanced deployment options
    DEPLOYMENT_OPTIONS = {
        'Single-AZ': 1,
        'Multi-AZ': 2,
        'Multi-AZ Cluster': 2.5,
        'Aurora Global': 3,
        'Serverless': 0.5  # Added serverless option
    }
    
    # Storage types with performance characteristics
    STORAGE_TYPES = {
        'gp2': {'iops_base': 100, 'throughput_base': 128},
        'gp3': {'iops_max': 16000, 'throughput_max': 1000},
        'io1': {'iops_ratio': 50, 'iops_max': 64000},
        'io2': {'iops_ratio': 500, 'iops_max': 256000}
    }
    
    # Enhanced environment profiles
    ENV_PROFILES = {
        "PROD": {
            "cpu_ram": 1.0, 
            "storage": 1.0,
            "ha_multiplier": 1.5,
            "backup_retention": 35,
            "maintenance_window": "sat:03:00-sat:04:00"
        },
        "SQA": {
            "cpu_ram": 0.75, 
            "storage": 0.7,
            "ha_multiplier": 1.2,
            "backup_retention": 14,
            "maintenance_window": "sun:01:00-sun:02:00"
        },
        "QA": {
            "cpu_ram": 0.6, 
            "storage": 0.5,
            "ha_multiplier": 1.0,
            "backup_retention": 7,
            "maintenance_window": "sun:00:00-sun:01:00"
        },
        "DEV": {
            "cpu_ram": 0.4, 
            "storage": 0.3,
            "ha_multiplier": 1.0,
            "backup_retention": 1,
            "maintenance_window": "any"
        }
    }
    
    # RI discount tiers
    RI_DISCOUNTS = {
        'No Upfront': {'1yr': 0.3, '3yr': 0.4},
        'Partial Upfront': {'1yr': 0.35, '3yr': 0.45},
        'All Upfront': {'1yr': 0.4, '3yr': 0.55}
    }
    
     def __init__(self):
        # Load instance database
        with open('instance_database.json') as f:
            self.INSTANCE_DB = json.load(f)
        
        # Initialize pricing data with default values
        self.pricing_data = {
            "storage": {
                "us-east-1": {
                    "gp2": 0.10,
                    "gp3": {
                        "gb": 0.08,
                        "iops": 0.005,
                        "throughput": 0.04
                    },
                    "io1": 0.125,
                    "io2": 0.15
                },
                "default": {
                    "gp2": 0.11,
                    "gp3": {
                        "gb": 0.09,
                        "iops": 0.006,
                        "throughput": 0.045
                    },
                    "io1": 0.135,
                    "io2": 0.16
                }
            },
            "backup": {
                "us-east-1": 0.05,
                "default": 0.055
            }
        }
        
        self.inputs = {
            "region": "us-east-1",
            "engine": "oracle-ee",
            "deployment": "Multi-AZ",
            "storage_type": "gp3",
            "on_prem_cores": 16,
            "peak_cpu_percent": 65,
            "on_prem_ram_gb": 64,
            "peak_ram_percent": 75,
            "storage_current_gb": 500,
            "storage_growth_rate": 0.15,
            "peak_iops": 8000,
            "peak_throughput_mbps": 400,
            "years": 3,
            "ha_replicas": 1,
            "backup_retention": 7,
            "enable_encryption": True,
            "enable_perf_insights": True,
            "enable_auto_scaling": False,
            "monthly_data_transfer_gb": 100,
            "ri_term": "No Upfront",
            "ri_duration": "1yr",
            "deployment_model": "Provisioned"  # Added deployment model
        }
        self.recommendations = {}
        self.tco_data = {}

    def _calculate_ram(self, env):
        """Enhanced RAM calculation with workload profiles"""
        base_ram = self.inputs["on_prem_ram_gb"] * (self.inputs["peak_ram_percent"] / 100)
        
        # Engine-specific multipliers
        engine_factors = {
            "oracle-ee": 0.85,
            "oracle-se": 0.8,
            "postgres": 0.9,
            "aurora-postgresql": 1.05,
            "aurora-mysql": 1.1,
            "sqlserver": 0.75
        }
        
        # Environment factor
        env_factor = self.ENV_PROFILES[env]["cpu_ram"]
        
        return max(4, math.ceil(base_ram * engine_factors[self.inputs["engine"]] * env_factor))
        
    def calculate_requirements(self, env):
        """Calculate requirements with enhanced optimization logic"""
        profile = self.ENV_PROFILES[env]
        
        # Compute requirements with workload-specific scaling
        vcpus = self._calculate_cpu(env)
        ram = self._calculate_ram(env)
        
        # Storage with compression estimation
        storage = self._calculate_storage(env)
        
        # IOPS with burst credit consideration
        iops = self._calculate_iops(env)
        
        # Select optimal instance
        instance = self._select_instance(vcpus, ram, iops, env)
        
        # Calculate costs
        costs = self._calculate_costs(instance, storage, iops, env)
        
        # Generate advisories
        advisories = self._generate_advisories(instance, storage, iops, env)
        
        return {
            "environment": env,
            "instance_type": instance["type"],
            "vCPUs": vcpus,
            "RAM_GB": ram,
            "storage_GB": storage,
            "iops": iops,
            "throughput": min(iops * 0.256, 1000),  # MB/s based on AWS ratios
            "instance_cost": costs["instance"],
            "storage_cost": costs["storage"],
            "backup_cost": costs["backup"],
            "ha_cost": costs["ha"],
            "features_cost": costs["features"],
            "total_cost": costs["total"],
            "advisories": advisories,
            "tco_savings": costs["tco_savings"]
        }
    
    def _calculate_cpu(self, env):
        """Enhanced CPU calculation with workload profiles"""
        base_cpu = self.inputs["on_prem_cores"] * (self.inputs["peak_cpu_percent"] / 100)
        
        # Engine-specific multipliers
        engine_factors = {
            "oracle-ee": 0.9,
            "oracle-se": 0.85,
            "postgres": 0.95,
            "aurora-postgresql": 1.1,
            "aurora-mysql": 1.15,
            "sqlserver": 0.8
        }
        
        # Environment factor
        env_factor = self.ENV_PROFILES[env]["cpu_ram"]
        
        return max(2, math.ceil(base_cpu * engine_factors[self.inputs["engine"]] * env_factor))
    
    def _calculate_storage(self, env):
        """Storage with compression estimation and growth projection"""
        growth_factor = (1 + self.inputs["storage_growth_rate"]) ** self.inputs["years"]
        base_storage = self.inputs["storage_current_gb"] * growth_factor
        
        # Compression factors by engine
        compression_ratios = {
            "oracle-ee": 2.5,
            "oracle-se": 2.0,
            "postgres": 1.8,
            "aurora-postgresql": 2.2,
            "aurora-mysql": 2.3,
            "sqlserver": 1.5
        }
        
        compressed = base_storage / compression_ratios[self.inputs["engine"]]
        env_factor = self.ENV_PROFILES[env]["storage"]
        
        return max(20, math.ceil(compressed * 1.3 * env_factor))  # 30% buffer
    
    def _calculate_iops(self, env):
        """IOPS calculation with burst credit consideration"""
        base_iops = self.inputs["peak_iops"]
        
        # Storage type capabilities
        storage_type = self.inputs["storage_type"]
        if storage_type == "gp2":
            # gp2: 3 IOPS/GB burstable
            max_iops = self._calculate_storage(env) * 3
            return min(base_iops * 1.3, max_iops)
        else:
            return base_iops * 1.3  # 30% buffer
    
    def _select_instance(self, vcpus, ram, iops, env):
        """AI-powered instance selection with serverless support"""
        engine = self.inputs["engine"]
        region = self.inputs["region"]
        deployment_model = self.inputs["deployment_model"]
        
        # Handle serverless deployment
        if deployment_model == "Serverless":
            serverless_candidate = {
                "type": "db.serverless",
                "vCPU": 0,
                "memory": 0,
                "max_iops": 0,
                "network_perf": "Up to 10 Gbps",
                "pricing": {"ondemand": 0.12}
            }
            return serverless_candidate
        
        # Get region data or fallback to us-east-1
        region_data = self.INSTANCE_DB.get(region, self.INSTANCE_DB.get("us-east-1", {}))
        engine_data = region_data.get(engine)
        
        if not engine_data:
            # Try to find any matching engine
            for r in self.INSTANCE_DB.values():
                if engine in r:
                    engine_data = r[engine]
                    break
            if not engine_data:
                raise ValueError(f"No instances found for engine '{engine}' in any region")
        
        candidates = [
            inst for inst in engine_data 
            if inst["vCPU"] >= vcpus 
            and inst["memory"] >= ram
            and inst.get("max_iops", float('inf')) >= iops
            and inst["network_perf"] >= self.inputs["peak_throughput_mbps"] / 1000
        ]
        
        if not candidates:
            # Fallback to largest instance
            return max(engine_data, key=lambda x: x["vCPU"])
        
        # Cost-performance scoring
        def instance_score(instance):
            # Weighted score: 60% cost, 30% performance, 10% newest generation
            cost = instance["pricing"]["ondemand"]
            perf = instance["vCPU"] * instance["memory"]
            gen_factor = 1.0 if "6" in instance["type"] else 0.8
            
            return (1/cost) * 0.6 + perf * 0.3 + gen_factor * 0.1
        
        return max(candidates, key=instance_score)

    def _calculate_costs(self, instance, storage, iops, env):
        """Comprehensive cost calculation with serverless support"""
        deployment_model = self.inputs["deployment_model"]
        
        # Serverless cost calculation
        if deployment_model == "Serverless":
            # Base serverless cost (per ACU-hour)
            hourly = instance["pricing"]["ondemand"]
            monthly_instance = hourly * 24 * 30
            
            # Storage cost for Aurora Serverless
            storage_cost = storage * 0.10  # $0.10/GB-month
            
            # HA cost (serverless includes HA)
            ha_cost = monthly_instance * 0.2  # 20% for HA
            
            # Backup cost
            region = self.inputs["region"]
            backup_rate = self.pricing_data["backup"].get(
                region, 
                self.pricing_data["backup"].get("default", 0.05)
            )
            backup_cost = storage * backup_rate * (self.inputs["backup_retention"] / 30)
            
            # Feature costs
            features_cost = monthly_instance * 0.15  # 15% for features
            
            total_cost = monthly_instance + storage_cost + ha_cost + backup_cost + features_cost
            
            # TCO savings
            onprem_cost = self._calculate_onprem_tco()
            tco_savings = (onprem_cost - total_cost) / onprem_cost * 100
            
            return {
                "instance": monthly_instance,
                "storage": storage_cost,
                "backup": backup_cost,
                "ha": ha_cost,
                "features": features_cost,
                "total": total_cost,
                "tco_savings": tco_savings
            }
    
    def _generate_advisories(self, instance, storage, iops, env):
        """Generate expert optimization advisories"""
        advisories = []
        deployment_model = self.inputs["deployment_model"]
        
        # Serverless advisory
        if "aurora" in self.inputs["engine"] and deployment_model == "Provisioned":
            advisories.append("ℹ️ Consider Serverless deployment for variable workloads")
        
        # High availability check
        if env == "PROD" and self.inputs["deployment"] == "Single-AZ":
            advisories.append("🚨 Production environment should use Multi-AZ deployment")
        
        # Storage optimization
        storage_type = self.inputs["storage_type"]
        if iops > 16000 and storage_type == "gp2":
            advisories.append("💡 Upgrade to gp3 or io2 storage for high IOPS workloads")
        
        # Right-sizing check
        util_ratio = instance["vCPU"] / self._calculate_cpu(env)
        if util_ratio > 1.5:
            advisories.append(f"⚖️ Consider smaller instance type (current utilization estimate: {util_ratio*100:.0f}%)")
        
        # Backup retention
        if env == "PROD" and self.inputs["backup_retention"] < 7:
            advisories.append("🔒 Increase backup retention to at least 7 days for production")
        
        # Encryption
        if not self.inputs["enable_encryption"]:
            advisories.append("🔐 Enable encryption at rest for security compliance")
        
        return advisories
    
    def _calculate_onprem_tco(self):
        """Calculate on-premise TCO for comparison"""
        # Includes hardware, maintenance, power, cooling, etc.
        cores = self.inputs["on_prem_cores"]
        ram = self.inputs["on_prem_ram_gb"]
        storage = self.inputs["storage_current_gb"]
        
        # Enterprise hardware costs
        server_cost = (cores * 1500) + (ram * 100)  # $/core + $/GB
        storage_cost = storage * 5  # $/GB
        maintenance = (server_cost + storage_cost) * 0.18  # 18% annual
        
        # 3-year TCO
        return (server_cost + storage_cost) + (maintenance * 3)
    
    def generate_all_recommendations(self):
        """Generate recommendations for all environments"""
        self.recommendations = {}
        for env in self.ENV_PROFILES:
            try:
                self.recommendations[env] = self.calculate_requirements(env)
            except Exception as e:
                logging.error(f"Error generating recommendations for {env}: {str(e)}")
                self.recommendations[env] = {"error": str(e)}
        
        # Generate TCO comparison
        self._generate_tco_comparison()
        
        return self.recommendations
    
    def _generate_tco_comparison(self):
        """Generate 3-year TCO projection"""
        years = self.inputs["years"]
        tco_data = []
        
        for year in range(1, years+1):
            year_data = {"Year": year}
            
            # On-prem costs
            year_data["OnPrem"] = self._calculate_onprem_tco() * (year/3)  # Proportional
            
            # Cloud costs
            cloud_cost = 0
            for env in self.recommendations:
                # Skip environments with errors
                if "error" in self.recommendations[env]:
                    continue
                    
                # Scale costs by year (accounting for storage growth)
                growth_factor = (1 + self.inputs["storage_growth_rate"]) ** (year-1)
                env_cost = self.recommendations[env]["total_cost"] * growth_factor * 12
                
                # Only include production in year 1, add others in subsequent years
                if year == 1 and env != "PROD":
                    continue
                elif year > 1:
                    cloud_cost += env_cost
            
            year_data["Cloud"] = cloud_cost
            tco_data.append(year_data)
        
        self.tco_data = tco_data