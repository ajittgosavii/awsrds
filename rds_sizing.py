import math
import json
import logging
import boto3
from datetime import datetime
from functools import lru_cache
from botocore.exceptions import ClientError, NoCredentialsError

class EnhancedRDSSizingCalculator:
    """
    Enhanced RDS sizing calculator with separate reader/writer sizing for Multi-AZ deployments
    """
    
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
        },
        'Serverless': {
            'cost_multiplier': 0.5,
            'has_readers': False,
            'reader_count': 0,
            'description': 'Auto-scaling serverless deployment'
        }
    }
    
    # Read/Write workload patterns
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
    
    # Environment profiles with enhanced characteristics
    ENV_PROFILES = {
        "PROD": {
            "cpu_multiplier": 1.0,
            "ram_multiplier": 1.0,
            "storage_multiplier": 1.0,
            "performance_buffer": 1.25,
            "ha_multiplier": 1.5,
            "backup_retention": 35,
            "min_instance_class": "m5",
            "cost_priority": 0.3,
            "reader_sizing_factor": 1.0,  # Full sizing for prod readers
            "description": "Production environment with full resources and performance headroom"
        },
        "SQA": {
            "cpu_multiplier": 0.75,
            "ram_multiplier": 0.8,
            "storage_multiplier": 0.7,
            "performance_buffer": 1.15,
            "ha_multiplier": 1.2,
            "backup_retention": 14,
            "min_instance_class": "t3",
            "cost_priority": 0.5,
            "reader_sizing_factor": 0.8,  # Slightly smaller readers for SQA
            "description": "System QA environment with reduced but adequate resources"
        },
        "QA": {
            "cpu_multiplier": 0.5,
            "ram_multiplier": 0.6,
            "storage_multiplier": 0.5,
            "performance_buffer": 1.1,
            "ha_multiplier": 1.0,
            "backup_retention": 7,
            "min_instance_class": "t3",
            "cost_priority": 0.7,
            "reader_sizing_factor": 0.6,  # Smaller readers for QA
            "description": "Quality Assurance environment optimized for cost"
        },
        "DEV": {
            "cpu_multiplier": 0.25,
            "ram_multiplier": 0.35,
            "storage_multiplier": 0.3,
            "performance_buffer": 1.0,
            "ha_multiplier": 1.0,
            "backup_retention": 1,
            "min_instance_class": "t3",
            "cost_priority": 0.9,
            "reader_sizing_factor": 0.4,  # Much smaller readers for dev
            "description": "Development environment with minimal resources"
        }
    }
    
    # Instance families with characteristics
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
        
        # Default inputs - enhanced with read/write workload configuration
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
            "deployment_model": "Provisioned",
            "workload_pattern": "OLTP_BALANCED",  # NEW: Workload pattern
            "read_write_ratio": "60:40",  # NEW: Custom read/write ratio
            "connection_pooling": True,  # NEW: Connection pooling enabled
            "reader_scaling": "auto"  # NEW: Reader scaling configuration
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
        
        # Adjust pricing for different regions
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
            # Default fallback
            return 60, 40
    
    def calculate_requirements(self, env):
        """
        Enhanced requirements calculation with reader/writer sizing for Multi-AZ
        """
        profile = self.ENV_PROFILES[env]
        deployment_config = self.DEPLOYMENT_OPTIONS[self.inputs["deployment"]]
        
        print(f"\n🔍 Calculating requirements for {env} environment:")
        print(f"   Profile: {profile['description']}")
        print(f"   Deployment: {self.inputs['deployment']} ({deployment_config['description']})")
        
        # Step 1: Calculate base resource requirements
        base_cpu_cores = self.inputs["on_prem_cores"] * (self.inputs["peak_cpu_percent"] / 100)
        base_ram_gb = self.inputs["on_prem_ram_gb"] * (self.inputs["peak_ram_percent"] / 100)
        
        print(f"   Base requirements: {base_cpu_cores:.1f} cores, {base_ram_gb:.1f}GB RAM")
        
        # Step 2: Apply environment-specific multipliers
        env_cpu_requirement = base_cpu_cores * profile["cpu_multiplier"] * profile["performance_buffer"]
        env_ram_requirement = base_ram_gb * profile["ram_multiplier"] * profile["performance_buffer"]
        
        # Step 3: Calculate read/write workload distribution
        read_pct, write_pct = self._parse_read_write_ratio()
        print(f"   Workload split: {read_pct:.0f}% reads, {write_pct:.0f}% writes")
        
        # Step 4: Calculate writer requirements (handles all writes + some reads)
        writer_cpu_requirement = env_cpu_requirement * (write_pct / 100 + 0.3 * read_pct / 100)  # Writer handles 30% of reads too
        writer_ram_requirement = env_ram_requirement * (write_pct / 100 + 0.3 * read_pct / 100)
        
        # Step 5: Apply environment minimums for writer
        env_minimums = {
            "PROD": {"cpu": 4, "ram": 8},
            "SQA": {"cpu": 2, "ram": 4},
            "QA": {"cpu": 2, "ram": 4},
            "DEV": {"cpu": 1, "ram": 2}
        }
        
        min_reqs = env_minimums[env]
        final_writer_cpu = max(math.ceil(writer_cpu_requirement), min_reqs["cpu"])
        final_writer_ram = max(math.ceil(writer_ram_requirement), min_reqs["ram"])
        
        # Step 6: Calculate reader requirements (if Multi-AZ deployment)
        reader_recommendations = {}
        if deployment_config["has_readers"]:
            # Reader handles remaining read workload
            remaining_read_workload = 0.7 * read_pct / 100  # 70% of reads go to readers
            reader_cpu_per_instance = env_cpu_requirement * remaining_read_workload / deployment_config["reader_count"]
            reader_ram_per_instance = env_ram_requirement * remaining_read_workload / deployment_config["reader_count"]
            
            # Apply reader sizing factor from environment profile
            reader_cpu_per_instance *= profile["reader_sizing_factor"]
            reader_ram_per_instance *= profile["reader_sizing_factor"]
            
            # Apply minimums for readers (can be smaller than writers)
            reader_min_cpu = max(1, min_reqs["cpu"] // 2) if env != "PROD" else min_reqs["cpu"]
            reader_min_ram = max(1, min_reqs["ram"] // 2) if env != "PROD" else min_reqs["ram"]
            
            final_reader_cpu = max(math.ceil(reader_cpu_per_instance), reader_min_cpu)
            final_reader_ram = max(math.ceil(reader_ram_per_instance), reader_min_ram)
            
            print(f"   Reader requirements per instance: {final_reader_cpu} vCPUs, {final_reader_ram}GB RAM")
            print(f"   Number of readers: {deployment_config['reader_count']}")
        
        # Step 7: Get available instances
        available_instances = self.get_instance_pricing_data(self.inputs["region"], self.inputs["engine"])
        
        # Step 8: Select optimal instances
        writer_instance = self._select_optimal_instance(
            final_writer_cpu, final_writer_ram, env, profile, available_instances, "writer"
        )
        
        print(f"   Selected Writer: {writer_instance['type']} ({writer_instance['vCPU']} vCPUs, {writer_instance['memory']}GB)")
        
        # Select reader instances if needed
        if deployment_config["has_readers"]:
            reader_instance = self._select_optimal_instance(
                final_reader_cpu, final_reader_ram, env, profile, available_instances, "reader"
            )
            
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
            
            print(f"   Selected Reader: {reader_instance['type']} ({reader_instance['vCPU']} vCPUs, {reader_instance['memory']}GB) x{deployment_config['reader_count']}")
        
        # Step 9: Calculate storage and IOPS
        storage_gb = self._calculate_storage_requirement(env, profile)
        iops_requirement = self._calculate_iops_requirement(env, profile)
        
        # Step 10: Calculate comprehensive costs
        costs = self._calculate_comprehensive_costs_with_readers(
            writer_instance, reader_recommendations, storage_gb, env, profile
        )
        
        # Step 11: Generate advisories
        advisories = self._generate_environment_advisories_with_readers(
            writer_instance, reader_recommendations, final_writer_cpu, final_writer_ram, env, profile
        )
        
        # Build comprehensive recommendation
        recommendation = {
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
            "readers": reader_recommendations,
            
            # Legacy fields for backward compatibility
            "instance_type": writer_instance["type"],
            "vCPUs": final_writer_cpu,
            "RAM_GB": final_writer_ram,
            "actual_vCPUs": writer_instance["vCPU"],
            "actual_RAM_GB": writer_instance["memory"],
            
            # Infrastructure details
            "storage_GB": storage_gb,
            "iops": iops_requirement,
            
            # Cost breakdown
            "instance_cost": costs["writer_monthly"],
            "reader_cost": costs.get("reader_monthly", 0),
            "storage_cost": costs["storage_monthly"],
            "backup_cost": costs["backup_monthly"],
            "total_cost": costs["total_monthly"],
            
            # Additional information
            "advisories": advisories,
            "cost_breakdown": costs,
            "profile_applied": profile,
            "has_readers": deployment_config["has_readers"]
        }
        
        return recommendation
    
    def _select_optimal_instance(self, cpu_req, ram_req, env, profile, available_instances, instance_role="writer"):
        """
        Enhanced instance selection with role-specific optimization
        """
        print(f"🎯 Selecting {instance_role} instance for {env}: need {cpu_req} vCPUs, {ram_req}GB RAM")
        
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
        
        # For readers in non-prod environments, allow smaller instance families
        if instance_role == "reader" and env in ["DEV", "QA"]:
            min_priority = 1  # Allow t3 instances for readers
        
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
        
        # Step 2: If no suitable instances found, relax requirements
        if not suitable_instances:
            tolerance = {
                "PROD": 0.95,  # 95% tolerance for production
                "SQA": 0.8,    # 80% tolerance for SQA
                "QA": 0.7,     # 70% tolerance for QA  
                "DEV": 0.5     # 50% tolerance for dev
            }
            
            # Readers can have more tolerance than writers
            if instance_role == "reader":
                tolerance = {k: v * 0.8 for k, v in tolerance.items()}
            
            env_tolerance = tolerance.get(env, 0.9)
            print(f"⚠️ No exact matches, applying {env_tolerance} tolerance for {env} {instance_role}")
            
            for instance in available_instances:
                meets_cpu = instance["vCPU"] >= cpu_req * env_tolerance
                meets_ram = instance["memory"] >= ram_req * env_tolerance
                
                if meets_cpu and meets_ram:
                    suitable_instances.append(instance)
        
        # Step 3: If still no matches, select closest available
        if not suitable_instances:
            print(f"⚠️ No suitable instances found, selecting best available for {instance_role}")
            suitable_instances = available_instances
        
        # Step 4: Apply role-specific selection strategy
        cost_priority = profile["cost_priority"]
        
        # Readers can prioritize cost more than writers
        if instance_role == "reader":
            cost_priority = min(cost_priority + 0.2, 1.0)
        
        def calculate_instance_score(instance):
            """Calculate instance score based on environment priorities and role"""
            
            # Resource efficiency (how well it matches requirements)
            cpu_ratio = instance["vCPU"] / max(cpu_req, 1)
            ram_ratio = instance["memory"] / max(ram_req, 1)
            
            # Penalize over-provisioning (more aggressive for readers)
            cpu_waste = max(0, cpu_ratio - 1.0)
            ram_waste = max(0, ram_ratio - 1.0)
            waste_penalty_factor = 0.7 if instance_role == "reader" else 0.5
            waste_penalty = (cpu_waste + ram_waste) * waste_penalty_factor
            
            # Cost factor
            cost_factor = 1.0 / (1.0 + instance["pricing"]["ondemand"])
            
            # Performance factor
            if env == "PROD" and instance_role == "writer":
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
        
        print(f"✅ Selected {best_instance['type']} for {env} {instance_role}")
        
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
    
    def _calculate_comprehensive_costs_with_readers(self, writer_instance, reader_recommendations, storage_gb, env, profile):
        """Calculate comprehensive monthly costs including readers"""
        
        # Writer instance cost
        writer_hourly_rate = writer_instance["pricing"]["ondemand"]
        writer_monthly = writer_hourly_rate * 24 * 30
        
        # Reader instance costs
        reader_monthly = 0
        if reader_recommendations:
            reader_monthly = reader_recommendations.get("total_reader_cost", 0)
        
        # Storage cost (shared across all instances)
        storage_cost_per_gb = {
            "gp2": 0.10,
            "gp3": 0.08,
            "io1": 0.125,
            "io2": 0.125
        }
        storage_rate = storage_cost_per_gb.get(self.inputs["storage_type"], 0.10)
        storage_monthly = storage_gb * storage_rate
        
        # Backup cost
        backup_retention_days = profile["backup_retention"]
        backup_monthly = storage_gb * 0.095 * (backup_retention_days / 30)
        
        # Additional features cost
        features_cost = 0
        total_instance_cost = writer_monthly + reader_monthly
        
        if self.inputs["enable_perf_insights"]:
            features_cost += total_instance_cost * 0.1
        
        if self.inputs["enable_encryption"]:
            features_cost += total_instance_cost * 0.02
        
        # Data transfer cost
        data_transfer_cost = self.inputs["monthly_data_transfer_gb"] * 0.09
        
        total_monthly = writer_monthly + reader_monthly + storage_monthly + backup_monthly + features_cost + data_transfer_cost
        
        return {
            "writer_monthly": writer_monthly,
            "reader_monthly": reader_monthly,
            "total_instance_monthly": total_instance_cost,
            "storage_monthly": storage_monthly,
            "backup_monthly": backup_monthly,
            "features_monthly": features_cost,
            "data_transfer_monthly": data_transfer_cost,
            "total_monthly": total_monthly,
            "tco_savings": 25  # Placeholder
        }
    
    def _generate_environment_advisories_with_readers(self, writer_instance, reader_recommendations, writer_cpu_req, writer_ram_req, env, profile):
        """Generate environment-specific optimization advisories including reader considerations"""
        advisories = []
        
        # Writer over-provisioning check
        writer_cpu_ratio = writer_instance["vCPU"] / max(writer_cpu_req, 1)
        writer_ram_ratio = writer_instance["memory"] / max(writer_ram_req, 1)
        
        if writer_cpu_ratio > 2:
            advisories.append(f"⚠️ Writer CPU over-provisioned: {writer_instance['vCPU']} vCPUs vs {writer_cpu_req} required")
        
        if writer_ram_ratio > 2:
            advisories.append(f"⚠️ Writer RAM over-provisioned: {writer_instance['memory']}GB vs {writer_ram_req}GB required")
        
        # Reader-specific advisories
        if reader_recommendations:
            reader_count = reader_recommendations["count"]
            reader_instance_type = reader_recommendations["instance_type"]
            
            if reader_count > 2 and env in ["DEV", "QA"]:
                advisories.append(f"💡 Consider reducing readers to 1 for {env} environment to save costs")
            
            if reader_recommendations["actual_vCPUs"] > writer_instance["vCPU"]:
                advisories.append("🔄 Reader instances are larger than writer - consider rebalancing")
            
            # Aurora-specific advisories
            if "aurora" in self.inputs["engine"]:
                advisories.append("🚀 Consider Aurora Auto Scaling for readers based on CPU utilization")
                
                if env in ["DEV", "QA"]:
                    advisories.append("💡 Aurora Serverless v2 might be cost-effective for variable dev/test workloads")
        
        # Environment-specific advisories
        if env == "PROD":
            if self.inputs["deployment"] == "Single-AZ":
                advisories.append("🚨 Production should use Multi-AZ deployment for high availability")
            
            if not reader_recommendations or reader_recommendations["count"] < 1:
                advisories.append("📖 Consider adding read replicas to offload read traffic from the writer")
            
            if profile["backup_retention"] < 7:
                advisories.append("🔒 Consider increasing backup retention for production compliance")
        
        elif env in ["DEV", "QA"]:
            total_cost = writer_instance["pricing"]["ondemand"] * 24 * 30
            if reader_recommendations:
                total_cost += reader_recommendations["total_reader_cost"]
            
            if total_cost > 500:  # $500/month threshold
                advisories.append(f"💰 Consider smaller instances or Single-AZ deployment for {env} environment")
            
            if reader_recommendations and reader_recommendations["count"] > 1:
                advisories.append(f"💡 Single read replica may be sufficient for {env} environment")
        
        # Workload pattern advisories
        read_pct, write_pct = self._parse_read_write_ratio()
        
        if read_pct > 80 and not reader_recommendations:
            advisories.append("📊 High read workload detected - consider adding read replicas")
        
        if write_pct > 70 and reader_recommendations and reader_recommendations["count"] > 1:
            advisories.append("✏️ Write-heavy workload detected - focus resources on writer instance")
        
        # Connection pooling recommendations
        if not self.inputs.get("connection_pooling", False):
            advisories.append("🔌 Enable connection pooling (RDS Proxy) to improve connection efficiency")
        
        return advisories
    
    def generate_all_recommendations(self):
        """Generate recommendations for all environments with reader/writer analysis"""
        print("\n🚀 Generating enhanced environment recommendations with reader/writer sizing...")
        print(f"Engine: {self.inputs['engine']}, Region: {self.inputs['region']}")
        print(f"Deployment: {self.inputs['deployment']}")
        print(f"Base workload: {self.inputs['on_prem_cores']} cores, {self.inputs['on_prem_ram_gb']}GB RAM")
        
        deployment_config = self.DEPLOYMENT_OPTIONS[self.inputs["deployment"]]
        if deployment_config["has_readers"]:
            print(f"📖 Multi-AZ deployment will include {deployment_config['reader_count']} reader instance(s)")
        else:
            print("📝 Single-AZ deployment - no reader instances")
        
        self.recommendations = {}
        
        for env in self.ENV_PROFILES:
            try:
                print(f"\n" + "="*60)
                recommendation = self.calculate_requirements(env)
                self.recommendations[env] = recommendation
                
                # Log the recommendation
                rec = recommendation
                print(f"✅ {env} Complete:")
                print(f"   Writer: {rec['writer']['instance_type']} (${rec['writer']['monthly_cost']:,.2f}/month)")
                
                if rec.get('readers'):
                    readers = rec['readers']
                    print(f"   Readers: {readers['instance_type']} x{readers['count']} (${readers['total_reader_cost']:,.2f}/month)")
                
                print(f"   Total Cost: ${rec['total_cost']:,.2f}/month")
                
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
        
        # Check writer instance type diversity
        writer_types = [r['writer']['instance_type'] for r in valid_recs.values()]
        unique_writer_types = len(set(writer_types))
        
        # Check reader instance diversity (if applicable)
        reader_types = [r['readers']['instance_type'] for r in valid_recs.values() if r.get('readers')]
        unique_reader_types = len(set(reader_types)) if reader_types else 0
        
        print(f"\n📊 Enhanced Recommendation Diversity Analysis:")
        print(f"   Environments: {len(valid_recs)}")
        print(f"   Unique writer types: {unique_writer_types}")
        if reader_types:
            print(f"   Unique reader types: {unique_reader_types}")
        
        if unique_writer_types == 1:
            print("⚠️ WARNING: All environments received the same writer instance type!")
        else:
            print("✅ Good diversity: Different environments got different writer instance types")
        
        # Cost progression check
        costs = {env: rec['total_cost'] for env, rec in valid_recs.items()}
        sorted_by_cost = sorted(costs.items(), key=lambda x: x[1])
        
        print(f"   Cost progression: {' < '.join([f'{env}(${cost:.0f})' for env, cost in sorted_by_cost])}")
        
        # Reader/Writer cost breakdown
        has_readers = any(r.get('readers') for r in valid_recs.values())
        if has_readers:
            print(f"\n📖 Reader/Writer Cost Analysis:")
            for env, rec in valid_recs.items():
                writer_cost = rec['writer']['monthly_cost']
                reader_cost = rec.get('readers', {}).get('total_reader_cost', 0)
                total_cost = writer_cost + reader_cost
                
                if reader_cost > 0:
                    reader_pct = (reader_cost / total_cost) * 100
                    print(f"   {env}: Writer ${writer_cost:.0f} ({100-reader_pct:.0f}%), Readers ${reader_cost:.0f} ({reader_pct:.0f}%)")
                else:
                    print(f"   {env}: Writer ${writer_cost:.0f} (100%), No readers")


# Testing and demonstration
if __name__ == "__main__":
    print("🧪 Testing Enhanced RDS Sizing Calculator with Reader/Writer Logic")
    
    # Initialize calculator
    calculator = EnhancedRDSSizingCalculator(use_real_time_pricing=True)
    
    # Test scenario 1: Multi-AZ PostgreSQL workload
    print("\n" + "="*80)
    print("TEST 1: Multi-AZ PostgreSQL workload with read replicas")
    calculator.inputs.update({
        "engine": "postgres",
        "deployment": "Multi-AZ",
        "workload_pattern": "READ_HEAVY",
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
            print(f"  {env}:")
            print(f"    Writer: {result['writer']['instance_type']} - ${result['writer']['monthly_cost']:,.2f}/month")
            if result.get('readers'):
                readers = result['readers']
                print(f"    Readers: {readers['instance_type']} x{readers['count']} - ${readers['total_reader_cost']:,.2f}/month")
            print(f"    Total: ${result['total_cost']:,.2f}/month")
        else:
            print(f"  {env}: ERROR - {result['error']}")
    
    # Test scenario 2: Aurora cluster with high read workload
    print("\n" + "="*80)
    print("TEST 2: Aurora Multi-AZ Cluster with read-heavy workload")
    calculator.inputs.update({
        "engine": "aurora-postgresql",
        "deployment": "Multi-AZ Cluster",
        "workload_pattern": "READ_HEAVY",
        "read_write_ratio": "85:15",
        "on_prem_cores": 24,
        "peak_cpu_percent": 80,
        "on_prem_ram_gb": 96,
        "peak_ram_percent": 85
    })
    
    results2 = calculator.generate_all_recommendations()
    
    print("\nTest 2 Results Summary:")
    for env, result in results2.items():
        if 'error' not in result:
            print(f"  {env}:")
            print(f"    Writer: {result['writer']['instance_type']} - ${result['writer']['monthly_cost']:,.2f}/month")
            if result.get('readers'):
                readers = result['readers']
                print(f"    Readers: {readers['instance_type']} x{readers['count']} - ${readers['total_reader_cost']:,.2f}/month")
            print(f"    Total: ${result['total_cost']:,.2f}/month")
        else:
            print(f"  {env}: ERROR - {result['error']}")
    
    # Test scenario 3: Single-AZ deployment (no readers)
    print("\n" + "="*80)
    print("TEST 3: Single-AZ deployment (no readers for comparison)")
    calculator.inputs.update({
        "engine": "postgres",
        "deployment": "Single-AZ",
        "workload_pattern": "MIXED",
        "on_prem_cores": 8,
        "peak_cpu_percent": 60,
        "on_prem_ram_gb": 32,
        "peak_ram_percent": 70
    })
    
    results3 = calculator.generate_all_recommendations()
    
    print("\nTest 3 Results Summary:")
    for env, result in results3.items():
        if 'error' not in result:
            print(f"  {env}: {result['writer']['instance_type']} - ${result['total_cost']:,.2f}/month (Single-AZ)")
        else:
            print(f"  {env}: ERROR - {result['error']}")
    
    print(f"\n🎯 Key Enhancements Applied:")
    print("✅ Separate writer and reader instance sizing for Multi-AZ deployments")
    print("✅ Workload pattern analysis (read/write ratio consideration)")
    print("✅ Environment-specific reader scaling factors")
    print("✅ Cost optimization for reader instances in non-prod environments")
    print("✅ Reader-specific advisories and recommendations")
    print("✅ Enhanced cost breakdown showing writer vs reader costs")
    print("✅ Support for different deployment types (Single-AZ, Multi-AZ, Multi-AZ Cluster)")
    print("✅ Real-time AWS pricing integration with reader/writer differentiation")