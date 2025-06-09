"""
Configuration settings for AI Database Migration Studio
"""
import os

class Config:
    # Application Info
    APP_NAME = "AI Database Migration Studio"
    APP_VERSION = "2.0.0"

    # Claude AI Model Configuration
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")
    AI_MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", 2500))
    AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", 0.1))

    # Supported Database Engines and AWS Regions
    SUPPORTED_ENGINES = [
        'oracle-ee', 'oracle-se', 'postgres',
        'aurora-postgresql', 'aurora-mysql', 'sqlserver'
    ]

    SUPPORTED_REGIONS = [
        "us-east-1", "us-west-1", "us-west-2",
        "eu-west-1", "ap-southeast-1"
    ]

    # Cost Configuration (example pricing in USD per GB)
    BASE_STORAGE_COST_GB = 0.115
    BASE_BACKUP_COST_GB = 0.095
    BASE_TRANSFER_COST_GB = 0.09

    # Environment Resource Multipliers
    ENV_PROFILES = {
        "PROD":    {"cpu_factor": 1.0, "storage_factor": 1.0, "ha_required": True},
        "STAGING": {"cpu_factor": 0.8, "storage_factor": 0.7, "ha_required": True},
        "QA":      {"cpu_factor": 0.6, "storage_factor": 0.5, "ha_required": False},
        "DEV":     {"cpu_factor": 0.4, "storage_factor": 0.3, "ha_required": False}
    }
"""
Configuration settings for AI Database Migration Studio
"""
import os

class Config:
    # Application Info
    APP_NAME = "AI Database Migration Studio"
    APP_VERSION = "2.0.0"

    # Claude AI Model Configuration
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20240229")
    AI_MAX_TOKENS = int(os.getenv("AI_MAX_TOKENS", 2500))
    AI_TEMPERATURE = float(os.getenv("AI_TEMPERATURE", 0.1))

    # Supported Database Engines and AWS Regions
    SUPPORTED_ENGINES = [
        'oracle-ee', 'oracle-se', 'postgres',
        'aurora-postgresql', 'aurora-mysql', 'sqlserver'
    ]

    SUPPORTED_REGIONS = [
        "us-east-1", "us-west-1", "us-west-2",
        "eu-west-1", "ap-southeast-1"
    ]

    # Cost Configuration (example pricing in USD per GB)
    BASE_STORAGE_COST_GB = 0.115
    BASE_BACKUP_COST_GB = 0.095
    BASE_TRANSFER_COST_GB = 0.09

    # Environment Resource Multipliers
    ENV_PROFILES = {
        "PROD":    {"cpu_factor": 1.0, "storage_factor": 1.0, "ha_required": True},
        "STAGING": {"cpu_factor": 0.8, "storage_factor": 0.7, "ha_required": True},
        "QA":      {"cpu_factor": 0.6, "storage_factor": 0.5, "ha_required": False},
        "DEV":     {"cpu_factor": 0.4, "storage_factor": 0.3, "ha_required": False}
    }
