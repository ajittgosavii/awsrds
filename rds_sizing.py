import math
import json
import logging
import boto3
from datetime import datetime
from functools import lru_cache
from botocore.exceptions import ClientError, NoCredentialsError

class FixedRDSDatabaseSizingCalculator:
    """
    Fixed RDS sizing calculator that properly differentiates environments
    and integrates real-time AWS pricing
    """
    
    ENGINES = ['oracle-ee', 'oracle-se', 'postgres', 'aurora-postgresql', 'aurora-mysql', 'sqlserver']
    
    # Fixed deployment options
    DEPLOYMENT_OPTIONS = {
        'Single-AZ': 1,
        'Multi-AZ': 2,
        'Multi-AZ Cluster': 2.5,
        'Aurora Global': 3,
        'Serverless': 0.5
    }
    
    # CRITICAL FIX: Properly differentiated environment profiles
    ENV_PROFILES = {
        "PROD": {
            "cpu_multiplier": 1.0,      # 100% of calculated requirements
            "ram_multiplier": 1.0,      # 100% of calculated requirements
            "storage_multiplier": 1.0,  # 100% of calculated requirements
            "performance_buffer": 1.25, # 25% performance headroom
            "ha_multiplier": 1.5,
            "backup_retention": 35,
            "min_instance_class": "m5",  # Prefer general-purpose or better
            "cost_priority": 0.3,       # Lower cost priority (30%), prefer performance
            "description": "Production environment with full resources and performance headroom"
        },
        "SQA": {
            "cpu_multiplier": 0.75,     # 75% of PROD requirements
            "ram_multiplier": 0.8,      # 80% of PROD requirements  
            "storage_multiplier": 0.7,  # 70% of PROD requirements
            "performance_buffer": 1.15, # 15% performance headroom
            "ha_multiplier": 1.2,
            "backup_retention": 14,
            "min_instance_class": "t3",  # Allow burstable instances
            "cost_priority": 0.5,       # Balanced cost/performance (50%)
            "description": "System QA environment with reduced but adequate resources"
        },
        "QA": {
            "cpu_multiplier": 0.5,      # 50% of PROD requirements
            "ram_multiplier": 0.6,      # 60% of PROD requirements
            "storage_multiplier": 0.5,  # 50% of PROD requirements
            "performance_buffer": 1.1,  # 10% performance headroom
            "ha_multiplier": 1.0,
            "backup_retention": 7,
            "min_instance_class": "t3",  # Allow burstable instances
            "cost_priority": 0.7,       # Higher cost priority (70%)
            "description": "Quality Assurance environment optimized for cost"
        },
        "DEV": {
            "cpu_multiplier": 0.25,     # 25% of PROD requirements
            "ram_multiplier": 0.35,     # 35% of PROD requirements
            "storage_multiplier": 0.3,  # 30% of PROD requirements
            "performance_buffer": 1.0,  # No performance headroom
            "ha_multiplier": 1.0,
            "backup_retention": 1,
            "min_instance_class": "t3",  # Prefer burstable instances
            "cost_priority": 0.9,       # Highest cost priority (90%)
            "description": "Development environment with minimal resources"
        }
    }
    
    # Enhanced instance families with characteristics
    INSTANCE_FAMILIES = {
        "t3": {"type": "burstable", "cpu_ratio": 1.0, "memory_ratio": 1.0, "cost_factor": 0.7},
        "t4g": {"type": "burstable", "cpu_ratio": 1.1, "memory_ratio": 1.0, "cost_factor": 0.6},
        "m5": {"type": "general", "cpu_ratio": 1.0, "memory_ratio": 1.0, "cost_factor": 1.0},
        "m6i": {"type": "general", "cpu_ratio": 1.1, "memory_ratio": 1.0, "cost_factor": 1.1},
        "r5": {"type": "memory", "cpu_ratio": 1.0, "memory_ratio": 2.0, "cost_factor": 1.3},
        "r6g": {"type": "memory", "cpu_ratio": 1.1, "memory_ratio": 2.0, "cost_factor": 1.2},
        "c5": {"type": "compute", "cpu_ratio": 1.5, "memory_ratio": 0.5, "cost_factor": 0.9}
    }
    
    def __init__(self, use_real_time_pricing=True):
        self.use_real_time_pricing = use_real_time_pricing
        self.pricing_cache = {}
        self.instance_cache = {}
        
        # Initialize AWS clients
        self.aws_available = self._initialize_aws_clients()
        
        # Default inputs
        self.inputs = {
            "region": "us-east-1",
            "engine": "postgres",
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
            "monthly_data_transfer_gb": 100,
            "ri_term": "No Upfront",
            "ri_duration": "1yr",
            "deployment_model": "Provisioned"
        }
        
        self.recommendations = {}
    
    def _initialize_aws_clients(self):
        """Initialize AWS clients for real-time pricing"""
        try:
            self.pricing_client = boto3.client('pricing', region_name='us-east-1')
            self.rds_client = boto3.client('rds', region_name='us-east-1')
            
            # Test the connection
            self.pricing_client.describe_services(ServiceCode='AmazonRDS', MaxResults=1)
            
            print("✅ AWS clients initialized successfully")
            return True
            
        except (NoCredentialsError, Exception) as e:
            print(f"⚠️ AWS not available: {e}")
            print("📝 Using fallback pricing data")
            self.pricing_client = None
            self.rds_client = None
            return False
    
    def get_instance_pricing_data(self, region, engine):
        """Get real-time or fallback instance pricing data"""
        cache_key = f"{region}_{engine}"
        
        if self.aws_available and self.use_real_time_pricing:
            return self._fetch_real_time_pricing(region, engine, cache_key)
        else:
            return self._get_fallback_pricing(region, engine)
    
    def _fetch_real_time_pricing(self, region, engine, cache_key):
        """Fetch real-time pricing from AWS Pricing API"""
        # Check cache first (cache for 1 hour)
        if cache_key in self.instance_cache:
            cached_data = self.instance_cache[cache_key]
            if (datetime.now().timestamp() - cached_data['timestamp']) < 3600:
                print(f"💾 Using cached pricing for {engine} in {region}")
                return cached_data['data']
        
        try:
            print(f"🌐 Fetching real-time pricing for {engine} in {region}...")
            
            # Map engine names to AWS API format
            engine_mapping = {
                'oracle-ee': 'Oracle',
                'oracle-se': 'Oracle', 
                'postgres': 'PostgreSQL',
                'aurora-postgresql': 'Aurora PostgreSQL',
                'aurora-mysql': 'Aurora MySQL',
                'sqlserver': 'SQL Server'
            }
            
            aws_engine = engine_mapping.get(engine, 'PostgreSQL')
            
            # Build filters for pricing API
            filters = [
                {'Type': 'TERM_MATCH', 'Field': 'regionCode', 'Value': region},
                {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': aws_engine},
                {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': 'Single-AZ'},
            ]
            
            # Add Oracle-specific filters
            if engine.startswith('oracle'):
                edition = 'Enterprise' if 'ee' in engine else 'Standard'
                filters.append({'Type': 'TERM_MATCH', 'Field': 'databaseEdition', 'Value': edition})
            
            instances = []
            next_token = None
            max_instances = 50  # Limit to prevent timeouts
            
            while len(instances) < max_instances:
                params = {
                    'ServiceCode': 'AmazonRDS',
                    'Filters': filters,
                    'MaxResults': 20
                }
                
                if next_token:
                    params['NextToken'] = next_token
                
                response = self.pricing_client.get_products(**params)
                
                for price_item in response['PriceList']:
                    try:
                        product = json.loads(price_item)
                        attributes = product['product']['attributes']
                        instance_type = attributes.get('instanceType')
                        
                        if instance_type and instance_type.startswith('db.'):
                            # Extract pricing information
                            terms = product['terms']['OnDemand']
                            price_dimension = next(iter(terms.values()))['priceDimensions']
                            price_per_hour = next(iter(price_dimension.values()))['pricePerUnit']['USD']
                            
                            # Create instance data structure
                            instance_data = {
                                "type": instance_type,
                                "vCPU": int(attributes.get('vcpu', '0')),
                                "memory": self._parse_memory(attributes.get('memory', '0 GiB')),
                                "max_iops": int(attributes.get('maxIops', '0')),
                                "network_performance": attributes.get('networkPerformance', 'Unknown'),
                                "pricing": {"ondemand": float(price_per_hour)},
                                "instance_family": instance_type.split('.')[1] if '.' in instance_type else 'unknown'
                            }
                            instances.append(instance_data)
                    
                    except (KeyError, ValueError, TypeError) as e:
                        logging.debug(f"Error parsing pricing item: {e}")
                        continue
                
                next_token = response.get('NextToken')
                if not next_token:
                    break
            
            if instances:
                # Sort instances by price for consistent ordering
                instances.sort(key=lambda x: x['pricing']['ondemand'])
                
                # Cache the results
                self.instance_cache[cache_key] = {
                    'data': instances,
                    'timestamp': datetime.now().timestamp()
                }
                
                print(f"✅ Fetched {len(instances)} instances for {engine}")
                return instances
            else:
                print(f"⚠️ No real-time pricing data found for {engine}, using fallback")
                return self._get_fallback_pricing(region, engine)
                
        except Exception as e:
            print(f"❌ Error fetching real-time pricing: {e}")
            return self._get_fallback_pricing(region, engine)
    
    def _parse_memory(self, memory_str):
        """Parse memory string like '8 GiB' to float"""
        try:
            if isinstance(memory_str, (int, float)):
                return float(memory_str)
            
            # Handle strings like "8 GiB", "16.0 GiB"
            parts = str(memory_str).split()
            if len(parts) >= 1:
                return float(parts[0])
            return 0.0
        except (ValueError, AttributeError):
            return 0.0
    
    def _get_fallback_pricing(self, region, engine):
        """Get fallback pricing data when AWS API is not available"""
        print(f"📝 Using fallback pricing for {engine} in {region}")
        
        # Enhanced fallback data with better instance variety
        fallback_data = {
            "postgres": [
                {"type": "db.t3.micro", "vCPU": 2, "memory": 1, "max_iops": 3000, "pricing": {"ondemand": 0.0255}, "instance_family": "t3"},
                {"type": "db.t3.small", "vCPU": 2, "memory": 2, "max_iops": 3000, "pricing": {"ondemand": 0.051}, "instance_family": "t3"},
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "max_iops": 3000, "pricing": {"ondemand": 0.102}, "instance_family": "t3"},
                {"type": "db.t3.large", "vCPU": 2, "memory": 8, "max_iops": 3000, "pricing": {"ondemand": 0.204}, "instance_family": "t3"},
                {"type": "db.m5.large", "vCPU": 2, "memory": 8, "max_iops": 7000, "pricing": {"ondemand": 0.192}, "instance_family": "m5"},
                {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "max_iops": 10000, "pricing": {"ondemand": 0.384}, "instance_family": "m5"},
                {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "max_iops": 15000, "pricing": {"ondemand": 0.768}, "instance_family": "m5"},
                {"type": "db.m5.4xlarge", "vCPU": 16, "memory": 64, "max_iops": 18750, "pricing": {"ondemand": 1.536}, "instance_family": "m5"},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "max_iops": 15000, "pricing": {"ondemand": 0.24}, "instance_family": "r5"},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "max_iops": 15000, "pricing": {"ondemand": 0.48}, "instance_family": "r5"},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "max_iops": 15000, "pricing": {"ondemand": 0.96}, "instance_family": "r5"},
            ],
            "oracle-ee": [
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "max_iops": 3000, "pricing": {"ondemand": 0.272}, "instance_family": "t3"},
                {"type": "db.t3.large", "vCPU": 2, "memory": 8, "max_iops": 3000, "pricing": {"ondemand": 0.544}, "instance_family": "t3"},
                {"type": "db.m5.large", "vCPU": 2, "memory": 8, "max_iops": 7000, "pricing": {"ondemand": 0.475}, "instance_family": "m5"},
                {"type": "db.m5.xlarge", "vCPU": 4, "memory": 16, "max_iops": 10000, "pricing": {"ondemand": 0.95}, "instance_family": "m5"},
                {"type": "db.m5.2xlarge", "vCPU": 8, "memory": 32, "max_iops": 15000, "pricing": {"ondemand": 1.90}, "instance_family": "m5"},
                {"type": "db.m5.4xlarge", "vCPU": 16, "memory": 64, "max_iops": 18750, "pricing": {"ondemand": 3.80}, "instance_family": "m5"},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "max_iops": 15000, "pricing": {"ondemand": 0.60}, "instance_family": "r5"},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "max_iops": 15000, "pricing": {"ondemand": 1.20}, "instance_family": "r5"},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "max_iops": 15000, "pricing": {"ondemand": 2.40}, "instance_family": "r5"},
            ],
            "aurora-postgresql": [
                {"type": "db.t3.medium", "vCPU": 2, "memory": 4, "max_iops": 3000, "pricing": {"ondemand": 0.082}, "instance_family": "t3"},
                {"type": "db.t4g.medium", "vCPU": 2, "memory": 4, "max_iops": 3000, "pricing": {"ondemand": 0.073}, "instance_family": "t4g"},
                {"type": "db.r5.large", "vCPU": 2, "memory": 16, "max_iops": 15000, "pricing": {"ondemand": 0.285}, "instance_family": "r5"},
                {"type": "db.r5.xlarge", "vCPU": 4, "memory": 32, "max_iops": 15000, "pricing": {"ondemand": 0.57}, "instance_family": "r5"},
                {"type": "db.r5.2xlarge", "vCPU": 8, "memory": 64, "max_iops": 15000, "pricing": {"ondemand": 1.14}, "instance_family": "r5"},
                {"type": "db.r6g.large", "vCPU": 2, "memory": 16, "max_iops": 15000, "pricing": {"ondemand": 0.256}, "instance_family": "r6g"},
                {"type": "db.r6g.xlarge", "vCPU": 4, "memory": 32, "max_iops": 15000, "pricing": {"ondemand": 0.512}, "instance_family": "r6g"},
                {"type": "db.serverless", "vCPU": 0, "memory": 0, "max_iops": 0, "pricing": {"ondemand": 0.12}, "instance_family": "serverless"},
            ]
        }
        
        # Default to postgres if engine not found
        engine_data = fallback_data.get(engine, fallback_data["postgres"])
        
        # Adjust pricing for different regions (rough estimates)
        region_multipliers = {
            "us-east-1": 1.0,
            "us-west-1": 1.08,
            "us-west-2": 1.08,
            "eu-west-1": 1.15,
            "ap-southeast-1": 1.20
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
    
    def calculate_requirements(self, env):
        """
        FIXED: Calculate requirements with proper environment differentiation
        """
        profile = self.ENV_PROFILES[env]
        
        print(f"\n🔍 Calculating requirements for {env} environment:")
        print(f"   Profile: {profile['description']}")
        
        # Step 1: Calculate base resource requirements
        base_cpu_cores = self.inputs["on_prem_cores"] * (self.inputs["peak_cpu_percent"] / 100)
        base_ram_gb = self.inputs["on_prem_ram_gb"] * (self.inputs["peak_ram_percent"] / 100)
        
        print(f"   Base requirements: {base_cpu_cores:.1f} cores, {base_ram_gb:.1f}GB RAM")
        
        # Step 2: Apply environment-specific multipliers
        env_cpu_requirement = base_cpu_cores * profile["cpu_multiplier"] * profile["performance_buffer"]
        env_ram_requirement = base_ram_gb * profile["ram_multiplier"] * profile["performance_buffer"]
        
        print(f"   Env multipliers: CPU {profile['cpu_multiplier']}, RAM {profile['ram_multiplier']}, Buffer {profile['performance_buffer']}")
        print(f"   Adjusted requirements: {env_cpu_requirement:.1f} cores, {env_ram_requirement:.1f}GB RAM")
        
        # Step 3: Apply environment minimums
        env_minimums = {
            "PROD": {"cpu": 4, "ram": 8},
            "SQA": {"cpu": 2, "ram": 4},
            "QA": {"cpu": 2, "ram": 4},
            "DEV": {"cpu": 1, "ram": 2}
        }
        
        min_reqs = env_minimums[env]
        final_cpu_requirement = max(math.ceil(env_cpu_requirement), min_reqs["cpu"])
        final_ram_requirement = max(math.ceil(env_ram_requirement), min_reqs["ram"])
        
        print(f"   Final requirements: {final_cpu_requirement} vCPUs, {final_ram_requirement}GB RAM")
        
        # Step 4: Calculate storage and IOPS
        storage_gb = self._calculate_storage_requirement(env, profile)
        iops_requirement = self._calculate_iops_requirement(env, profile)
        
        # Step 5: Get available instances
        available_instances = self.get_instance_pricing_data(self.inputs["region"], self.inputs["engine"])
        
        # Step 6: Select optimal instance (THIS IS THE KEY FIX)
        selected_instance = self._select_optimal_instance_fixed(
            final_cpu_requirement, final_ram_requirement, env, profile, available_instances
        )
        
        print(f"   Selected: {selected_instance['type']} ({selected_instance['vCPU']} vCPUs, {selected_instance['memory']}GB)")
        
        # Step 7: Calculate costs
        costs = self._calculate_comprehensive_costs(selected_instance, storage_gb, env, profile)
        
        # Step 8: Generate advisories
        advisories = self._generate_environment_advisories(
            selected_instance, final_cpu_requirement, final_ram_requirement, env, profile
        )
        
        return {
            "environment": env,
            "instance_type": selected_instance["type"],
            "vCPUs": final_cpu_requirement,
            "RAM_GB": final_ram_requirement,
            "actual_vCPUs": selected_instance["vCPU"],
            "actual_RAM_GB": selected_instance["memory"],
            "storage_GB": storage_gb,
            "iops": iops_requirement,
            "instance_cost": costs["instance_monthly"],
            "storage_cost": costs["storage_monthly"],
            "backup_cost": costs["backup_monthly"],
            "total_cost": costs["total_monthly"],
            "advisories": advisories,
            "cost_breakdown": costs,
            "tco_savings": costs.get("tco_savings", 0),
            "profile_applied": profile
        }
    
    def _select_optimal_instance_fixed(self, cpu_req, ram_req, env, profile, available_instances):
        """
        CRITICAL FIX: Proper instance selection that differentiates environments
        """
        print(f"🎯 Selecting instance for {env}: need {cpu_req} vCPUs, {ram_req}GB RAM")
        
        if not available_instances:
            raise ValueError(f"No instances available for {self.inputs['engine']} in {self.inputs['region']}")
        
        # Handle serverless deployment
        if self.inputs["deployment_model"] == "Serverless":
            serverless_instances = [i for i in available_instances if "serverless" in i["type"]]
            if serverless_instances:
                return serverless_instances[0]
        
        # Filter instances by minimum instance family requirement
        min_family = profile["min_instance_class"]
        family_priority = {"t3": 1, "t4g": 1, "m5": 2, "m6i": 2, "r5": 3, "r6g": 3, "c5": 2}
        min_priority = family_priority.get(min_family, 1)
        
        # Step 1: Filter instances that meet resource requirements
        suitable_instances = []
        for instance in available_instances:
            # Check resource requirements
            meets_cpu = instance["vCPU"] >= cpu_req
            meets_ram = instance["memory"] >= ram_req
            
            # Check instance family preference
            instance_family = instance.get("instance_family", instance["type"].split('.')[1] if '.' in instance["type"] else "unknown")
            instance_priority = family_priority.get(instance_family, 1)
            meets_family_req = instance_priority >= min_priority
            
            if meets_cpu and meets_ram and meets_family_req:
                suitable_instances.append(instance)
        
        # Step 2: If no suitable instances found, relax requirements for non-prod
        if not suitable_instances:
            tolerance = {
                "PROD": 0.95,  # 95% tolerance for production
                "SQA": 0.8,    # 80% tolerance for SQA
                "QA": 0.7,     # 70% tolerance for QA  
                "DEV": 0.5     # 50% tolerance for dev
            }
            
            env_tolerance = tolerance.get(env, 0.9)
            print(f"⚠️ No exact matches, applying {env_tolerance} tolerance for {env}")
            
            for instance in available_instances:
                meets_cpu = instance["vCPU"] >= cpu_req * env_tolerance
                meets_ram = instance["memory"] >= ram_req * env_tolerance
                
                if meets_cpu and meets_ram:
                    suitable_instances.append(instance)
        
        # Step 3: If still no matches, select closest available
        if not suitable_instances:
            print(f"⚠️ No suitable instances found, selecting best available")
            suitable_instances = available_instances
        
        # Step 4: Apply environment-specific selection strategy
        cost_priority = profile["cost_priority"]
        
        def calculate_instance_score(instance):
            """Calculate instance score based on environment priorities"""
            
            # Resource efficiency (how well it matches requirements)
            cpu_ratio = instance["vCPU"] / max(cpu_req, 1)
            ram_ratio = instance["memory"] / max(ram_req, 1)
            
            # Penalize over-provisioning
            cpu_waste = max(0, cpu_ratio - 1.0)
            ram_waste = max(0, ram_ratio - 1.0)
            waste_penalty = (cpu_waste + ram_waste) * 0.5
            
            # Cost factor
            cost_factor = 1.0 / (1.0 + instance["pricing"]["ondemand"])
            
            # Performance factor (prefer some headroom for production)
            if env == "PROD":
                performance_bonus = min(cpu_ratio + ram_ratio - 2.0, 1.0) * 0.3
            else:
                performance_bonus = 0
            
            # Instance family bonus
            family = instance.get("instance_family", "unknown")
            family_characteristics = self.INSTANCE_FAMILIES.get(family, {"cost_factor": 1.0})
            family_bonus = (1.0 / family_characteristics["cost_factor"]) * 0.1
            
            # Final score calculation
            efficiency_score = (2.0 - waste_penalty) * (1.0 - cost_priority)
            cost_score = cost_factor * cost_priority
            total_score = efficiency_score + cost_score + performance_bonus + family_bonus
            
            return total_score
        
        # Select the best instance based on scoring
        best_instance = max(suitable_instances, key=calculate_instance_score)
        
        print(f"✅ Selected {best_instance['type']} for {env}")
        print(f"   Score factors: Cost priority {cost_priority}, Family {best_instance.get('instance_family', 'unknown')}")
        
        return best_instance
    
    def _calculate_storage_requirement(self, env, profile):
        """Calculate storage requirements for environment"""
        base_storage = self.inputs["storage_current_gb"]
        growth_factor = (1 + self.inputs["storage_growth_rate"]) ** self.inputs["years"]
        projected_storage = base_storage * growth_factor
        
        # Apply environment multiplier
        env_storage = projected_storage * profile["storage_multiplier"]
        
        # Add buffer and ensure minimum
        storage_with_buffer = env_storage * 1.3  # 30% buffer
        min_storage = {"PROD": 100, "SQA": 50, "QA": 50, "DEV": 20}[env]
        
        return max(min_storage, math.ceil(storage_with_buffer))
    
    def _calculate_iops_requirement(self, env, profile):
        """Calculate IOPS requirements for environment"""
        base_iops = self.inputs["peak_iops"]
        env_iops = base_iops * profile["cpu_multiplier"] * profile["performance_buffer"]
        
        min_iops = {"PROD": 3000, "SQA": 2000, "QA": 1500, "DEV": 1000}[env]
        
        return max(min_iops, math.ceil(env_iops))
    
    def _calculate_comprehensive_costs(self, instance, storage_gb, env, profile):
        """Calculate comprehensive monthly costs"""
        
        # Instance cost
        hourly_rate = instance["pricing"]["ondemand"]
        deployment_factor = self.DEPLOYMENT_OPTIONS.get(self.inputs["deployment"], 1)
        monthly_instance = hourly_rate * 24 * 30 * deployment_factor
        
        # Storage cost (simplified)
        storage_cost_per_gb = {
            "gp2": 0.10,
            "gp3": 0.08,
            "io1": 0.125,
            "io2": 0.125
        }
        storage_rate = storage_cost_per_gb.get(self.inputs["storage_type"], 0.10)
        monthly_storage = storage_gb * storage_rate
        
        # Backup cost
        backup_retention_days = profile["backup_retention"]
        monthly_backup = storage_gb * 0.095 * (backup_retention_days / 30)
        
        # Additional features cost
        features_cost = 0
        if self.inputs["enable_perf_insights"]:
            features_cost += monthly_instance * 0.1
        
        if self.inputs["enable_encryption"]:
            features_cost += monthly_instance * 0.02
        
        # Data transfer cost
        data_transfer_cost = self.inputs["monthly_data_transfer_gb"] * 0.09
        
        total_monthly = monthly_instance + monthly_storage + monthly_backup + features_cost + data_transfer_cost
        
        return {
            "instance_monthly": monthly_instance,
            "storage_monthly": monthly_storage,
            "backup_monthly": monthly_backup,
            "features_monthly": features_cost,
            "data_transfer_monthly": data_transfer_cost,
            "total_monthly": total_monthly,
            "tco_savings": 25  # Placeholder
        }
    
    def _generate_environment_advisories(self, instance, cpu_req, ram_req, env, profile):
        """Generate environment-specific optimization advisories"""
        advisories = []
        
        # Over-provisioning check
        cpu_ratio = instance["vCPU"] / max(cpu_req, 1)
        ram_ratio = instance["memory"] / max(ram_req, 1)
        
        if cpu_ratio > 2:
            advisories.append(f"⚠️ CPU over-provisioned: {instance['vCPU']} vCPUs vs {cpu_req} required")
        
        if ram_ratio > 2:
            advisories.append(f"⚠️ RAM over-provisioned: {instance['memory']}GB vs {ram_req}GB required")
        
        # Environment-specific advisories
        if env == "PROD":
            if self.inputs["deployment"] == "Single-AZ":
                advisories.append("🚨 Production should use Multi-AZ deployment for high availability")
            
            if profile["backup_retention"] < 7:
                advisories.append("🔒 Consider increasing backup retention for production compliance")
        
        elif env in ["DEV", "QA"]:
            if instance["pricing"]["ondemand"] > 1.0:
                advisories.append("💡 Consider smaller instances for development/testing environments")
            
            if "aurora" in self.inputs["engine"]:
                advisories.append("💡 Consider Aurora Serverless for variable dev/test workloads")
        
        # Instance family advisories
        family = instance.get("instance_family", "unknown")
        if env == "DEV" and family in ["r5", "r6g"] and ram_ratio > 1.5:
            advisories.append("💰 Consider general-purpose instances instead of memory-optimized for development")
        
        return advisories
    
    def generate_all_recommendations(self):
        """Generate recommendations for all environments with proper differentiation"""
        print("\n🚀 Generating environment-differentiated recommendations...")
        print(f"Engine: {self.inputs['engine']}, Region: {self.inputs['region']}")
        print(f"Base workload: {self.inputs['on_prem_cores']} cores, {self.inputs['on_prem_ram_gb']}GB RAM")
        
        self.recommendations = {}
        
        for env in self.ENV_PROFILES:
            try:
                print(f"\n" + "="*50)
                recommendation = self.calculate_requirements(env)
                self.recommendations[env] = recommendation
                
                # Log the recommendation
                rec = recommendation
                print(f"✅ {env} Complete:")
                print(f"   Instance: {rec['instance_type']}")
                print(f"   Resources: {rec['actual_vCPUs']} vCPUs, {rec['actual_RAM_GB']}GB RAM")
                print(f"   Cost: ${rec['total_cost']:,.2f}/month")
                
            except Exception as e:
                print(f"❌ Error in {env}: {str(e)}")
                import traceback
                traceback.print_exc()
                self.recommendations[env] = {"error": str(e)}
        
        # Validate recommendations diversity
        self._validate_recommendations_diversity()
        
        return self.recommendations
    
    def _validate_recommendations_diversity(self):
        """Validate that environments have properly differentiated recommendations"""
        valid_recs = {k: v for k, v in self.recommendations.items() if 'error' not in v}
        
        if len(valid_recs) < 2:
            print("⚠️ Cannot validate diversity - insufficient valid recommendations")
            return
        
        # Check instance type diversity
        instance_types = [r['instance_type'] for r in valid_recs.values()]
        unique_types = len(set(instance_types))
        
        print(f"\n📊 Recommendation Diversity Analysis:")
        print(f"   Environments: {len(valid_recs)}")
        print(f"   Unique instance types: {unique_types}")
        
        if unique_types == 1:
            print("⚠️ WARNING: All environments received the same instance type!")
            print("   This suggests the instance selection logic needs review.")
        else:
            print("✅ Good diversity: Different environments got different instance types")
        
        # Cost progression check
        costs = {env: rec['total_cost'] for env, rec in valid_recs.items()}
        sorted_by_cost = sorted(costs.items(), key=lambda x: x[1])
        
        print(f"   Cost progression: {' < '.join([f'{env}(${cost:.0f})' for env, cost in sorted_by_cost])}")
        
        # Environment ordering check
        expected_order = ['DEV', 'QA', 'SQA', 'PROD']
        actual_order = [env for env, _ in sorted_by_cost]
        
        if actual_order == [env for env in expected_order if env in costs]:
            print("✅ Cost progression follows expected environment hierarchy")
        else:
            print("💡 Cost progression differs from typical hierarchy (may be acceptable)")

# Testing and demonstration
if __name__ == "__main__":
    print("🧪 Testing Fixed RDS Sizing Calculator")
    
    # Initialize calculator
    calculator = FixedRDSDatabaseSizingCalculator(use_real_time_pricing=True)
    
    # Test scenario 1: Standard workload
    print("\n" + "="*70)
    print("TEST 1: Standard PostgreSQL workload")
    calculator.inputs.update({
        "engine": "postgres",
        "on_prem_cores": 16,
        "peak_cpu_percent": 70,
        "on_prem_ram_gb": 64,
        "peak_ram_percent": 75,
        "region": "us-east-1"
    })
    
    results1 = calculator.generate_all_recommendations()
    
    print("\nTest 1 Results Summary:")
    for env, result in results1.items():
        if 'error' not in result:
            print(f"  {env}: {result['instance_type']} - ${result['total_cost']:,.2f}/month")
        else:
            print(f"  {env}: ERROR - {result['error']}")
    
    # Test scenario 2: High-end Oracle workload
    print("\n" + "="*70)
    print("TEST 2: High-end Oracle EE workload")
    calculator.inputs.update({
        "engine": "oracle-ee",
        "on_prem_cores": 32,
        "peak_cpu_percent": 80,
        "on_prem_ram_gb": 128,
        "peak_ram_percent": 85
    })
    
    results2 = calculator.generate_all_recommendations()
    
    print("\nTest 2 Results Summary:")
    for env, result in results2.items():
        if 'error' not in result:
            print(f"  {env}: {result['instance_type']} - ${result['total_cost']:,.2f}/month")
        else:
            print(f"  {env}: ERROR - {result['error']}")
    
    print(f"\n🎯 Key Fixes Applied:")
    print("✅ Environment-specific resource multipliers")
    print("✅ Proper instance selection logic with scoring")
    print("✅ Cost optimization strategies per environment")
    print("✅ Real-time AWS pricing integration")
    print("✅ Recommendation diversity validation")
    print("✅ Enhanced fallback pricing data")