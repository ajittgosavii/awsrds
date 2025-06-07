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
        'Serverless': 0.5
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
        # Replace static pricing with dynamic client
        from aws_pricing import AWSPricing
        self.pricing_client = AWSPricing()
        
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
            "deployment_model": "Provisioned"
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
        
        calculated_ram = max(4, math.ceil(base_ram * engine_factors[self.inputs["engine"]] * env_factor))
        
        # Ensure non-PROD environments have at least 4GB RAM
        if env != "PROD":
            return max(4, calculated_ram)
        return calculated_ram
        
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
            "data_transfer_cost": costs["data_transfer"],
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
        
        calculated_cpu = max(2, math.ceil(base_cpu * engine_factors[self.inputs["engine"]] * env_factor))
        
        # Ensure non-PROD environments have at least 2 vCPUs
        if env != "PROD":
            return max(2, calculated_cpu)
        return calculated_cpu

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
    
    def _get_max_iops(self, storage_type, storage_size):
        """Get maximum IOPS for storage type"""
        if storage_type == "gp2":
            return storage_size * 3
        elif storage_type == "gp3":
            return 16000
        elif storage_type == "io1":
            return 64000
        elif storage_type == "io2":
            return 256000
        return 0
    
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
        
        # Relaxed candidate filtering
        candidates = [
            inst for inst in engine_data 
            if inst["vCPU"] >= vcpus * 0.8  # Allow 20% underprovisioning
            and inst["memory"] >= ram * 0.8
            # Remove strict IOPS constraint
        ]
        
        if not candidates:
            # Find closest match by performance score
            def performance_score(inst):
                cpu_match = 1 - abs(inst["vCPU"] - vcpus) / max(vcpus, 1)
                ram_match = 1 - abs(inst["memory"] - ram) / max(ram, 1)
                return (cpu_match * 0.6) + (ram_match * 0.4)
                
            return max(engine_data, key=performance_score)
        
        # Cost-performance scoring with relaxed weights
        def instance_score(instance):
            cost = instance["pricing"]["ondemand"]
            perf = (instance["vCPU"] * 0.5) + (instance["memory"] * 0.5)
            return perf / cost  # Higher value = better value
        
        return min(candidates, key=lambda x: x["pricing"]["ondemand"])  # Select cheapest valid option

    def _calculate_costs(self, instance, storage, iops, env):
        """Comprehensive cost calculation for both serverless and provisioned"""
        region = self.inputs["region"]
        deployment_model = self.inputs["deployment_model"]
        
        # Get pricing data for region or use default
        storage_pricing = self.pricing_data["storage"].get(
            region, self.pricing_data["storage"]["default"]
        )
        backup_rate = self.pricing_data["backup"].get(
            region, self.pricing_data["backup"]["default"]
        )
        data_transfer_rate = self.pricing_data["data_transfer"].get(
            region, self.pricing_data["data_transfer"]["default"]
        )
        
        # Serverless cost calculation
        if deployment_model == "Serverless":
            # Get serverless pricing for engine
            serverless_pricing = self.pricing_data["serverless"].get(
                region, self.pricing_data["serverless"]["default"]
            ).get(self.inputs["engine"], {"acu": 0.12, "storage": 0.10})
            
            # Estimate ACU based on vCPU and RAM
            acu = max(self._calculate_cpu(env), math.ceil(self._calculate_ram(env) / 2))
            
            # Calculate costs
            monthly_instance = acu * serverless_pricing["acu"] * 24 * 30
            storage_cost = storage * serverless_pricing["storage"]
            backup_cost = storage * backup_rate * (self.inputs["backup_retention"] / 30)
            data_transfer_cost = self.inputs["monthly_data_transfer_gb"] * data_transfer_rate
            
            # Feature costs
            features_cost = monthly_instance * 0.15 if self.inputs["enable_perf_insights"] else 0
            
            # HA cost included in serverless
            ha_cost = monthly_instance * 0.2
            
            total_cost = monthly_instance + storage_cost + backup_cost + data_transfer_cost + features_cost + ha_cost
            
        else:  # Provisioned cost calculation
            # Instance cost
            base_hourly = instance["pricing"]["ondemand"]
            deployment_factor = self.DEPLOYMENT_OPTIONS[self.inputs["deployment"]]
            monthly_instance = base_hourly * 24 * 30 * deployment_factor
            
            # Replicas cost
            replicas_cost = base_hourly * 24 * 30 * self.inputs["ha_replicas"]
            
            # Storage cost
            if self.inputs["storage_type"] == "gp3":
                # GP3 includes 3000 IOPS and 125 MB/s free
                free_iops = 3000
                free_throughput = 125
                extra_iops = max(0, iops - free_iops)
                extra_throughput = max(0, self.inputs["peak_throughput_mbps"] - free_throughput)
                
                storage_cost = storage * storage_pricing["gp3"]["gb"]
                storage_cost += extra_iops * storage_pricing["gp3"]["iops"]
                storage_cost += extra_throughput * storage_pricing["gp3"]["throughput"]
            elif self.inputs["storage_type"] in ["io1", "io2"]:
                storage_cost = storage * storage_pricing[self.inputs["storage_type"]]["gb"]
                storage_cost += iops * storage_pricing[self.inputs["storage_type"]]["iops"]
            else:  # gp2
                storage_cost = storage * storage_pricing[self.inputs["storage_type"]]
            
            # Other costs
            backup_cost = storage * backup_rate * (self.inputs["backup_retention"] / 30)
            data_transfer_cost = self.inputs["monthly_data_transfer_gb"] * data_transfer_rate
            features_cost = monthly_instance * 0.15 if self.inputs["enable_perf_insights"] else 0
            
            # HA cost (already included in deployment factor)
            ha_cost = 0
            
            # Apply RI discount
            ri_discount = self.RI_DISCOUNTS[self.inputs["ri_term"]][self.inputs["ri_duration"]]
            discounted_instance = monthly_instance * (1 - ri_discount)
            
            total_cost = (discounted_instance + replicas_cost + storage_cost + 
                          backup_cost + data_transfer_cost + features_cost + ha_cost)
        
        # TCO savings
        onprem_cost = self._calculate_onprem_tco()
        tco_savings = (onprem_cost - total_cost) / onprem_cost * 100 if onprem_cost > 0 else 0
        
        return {
            "instance": monthly_instance if deployment_model == "Serverless" else discounted_instance + replicas_cost,
            "storage": storage_cost,
            "backup": backup_cost,
            "ha": ha_cost,
            "features": features_cost,
            "data_transfer": data_transfer_cost,
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
        max_iops = self._get_max_iops(storage_type, storage)
        if iops > max_iops:
            advisories.append(f"🚨 IOPS requirement ({iops}) exceeds {storage_type} max ({max_iops})")
        
        if storage_type == "gp3" and self.inputs["peak_throughput_mbps"] > 1000:
            advisories.append("🚨 Throughput requirement exceeds gp3 max (1000 MB/s)")
        
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
        # Hardware costs
        cores = self.inputs["on_prem_cores"]
        ram = self.inputs["on_prem_ram_gb"]
        storage = self.inputs["storage_current_gb"]
        
        server_cost = (cores * 1500) + (ram * 100)  # $/core + $/GB
        storage_cost = storage * 5  # $/GB
        
        # Maintenance (18% annual)
        maintenance = (server_cost + storage_cost) * 0.18 * self.inputs["years"]
        
        # Power, cooling, space (estimated)
        operational = (server_cost * 0.15) * self.inputs["years"]
        
        return server_cost + storage_cost + maintenance + operational
    
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
        """Generate multi-year TCO projection"""
        years = self.inputs["years"]
        tco_data = []
        onprem_costs = self._calculate_onprem_tco_breakdown()
        
        for year in range(1, years + 1):
            year_data = {"Year": year}
            
            # On-prem costs
            year_data["OnPrem"] = onprem_costs["initial"] + (onprem_costs["annual"] * year)
            
            # Cloud costs
            cloud_cost = 0
            for env in self.recommendations:
                if "error" in self.recommendations[env]:
                    continue
                    
                # Scale costs by year (accounting for storage growth)
                growth_factor = (1 + self.inputs["storage_growth_rate"]) ** (year - 1)
                env_cost = self.recommendations[env]["total_cost"] * growth_factor * 12
                
                # Only include production in year 1, add others in subsequent years
                if year == 1 and env != "PROD":
                    continue
                elif year > 1:
                    cloud_cost += env_cost
            
            year_data["Cloud"] = cloud_cost
            tco_data.append(year_data)
        
        self.tco_data = tco_data
    
    def _calculate_onprem_tco_breakdown(self):
        """Breakdown on-prem costs for TCO comparison"""
        cores = self.inputs["on_prem_cores"]
        ram = self.inputs["on_prem_ram_gb"]
        storage = self.inputs["storage_current_gb"]
        
        server_cost = (cores * 1500) + (ram * 100)
        storage_cost = storage * 5
        annual_maintenance = (server_cost + storage_cost) * 0.18
        
        return {
            "initial": server_cost + storage_cost,
            "annual": annual_maintenance
        }