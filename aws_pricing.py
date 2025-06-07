import boto3
import json
import time
from botocore.exceptions import ClientError

class AWSPricing:
    REGIONS = ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
    CACHE_DURATION = 86400  # 24 hours
    
    def __init__(self):
        self.cache = {}
        self.last_updated = {}
        self.client = boto3.client('pricing', region_name='us-east-1')
    
    def get_rds_pricing(self, region, engine):
        cache_key = f"{region}_{engine}"
        
        # Check cache
        if cache_key in self.cache and time.time() - self.last_updated.get(cache_key, 0) < self.CACHE_DURATION:
            return self.cache[cache_key]
        
        try:
            filters = [
                {'Type': 'TERM_MATCH', 'Field': 'regionCode', 'Value': region},
                {'Type': 'TERM_MATCH', 'Field': 'databaseEngine', 'Value': engine.split('-')[0].capitalize()},
                {'Type': 'TERM_MATCH', 'Field': 'deploymentOption', 'Value': 'Single-AZ'},
                {'Type': 'TERM_MATCH', 'Field': 'licenseModel', 'Value': 'License included'},
            ]
            
            if engine.startswith('oracle'):
                filters.append({'Type': 'TERM_MATCH', 'Field': 'databaseEdition', 'Value': engine.split('-')[1].upper()})
            
            prices = {}
            next_token = None
            
            while True:
                if next_token:
                    response = self.client.get_products(
                        ServiceCode='AmazonRDS',
                        Filters=filters,
                        NextToken=next_token
                    )
                else:
                    response = self.client.get_products(
                        ServiceCode='AmazonRDS',
                        Filters=filters
                    )
                
                for price_item in response['PriceList']:
                    product = json.loads(price_item)
                    instance_type = product['product']['attributes'].get('instanceType')
                    
                    if instance_type:
                        terms = product['terms']['OnDemand']
                        price_dim = next(iter(terms.values()))['priceDimensions']
                        price = next(iter(price_dim.values()))['pricePerUnit']['USD']
                        
                        # Extract additional attributes
                        attributes = product['product']['attributes']
                        instance_data = {
                            "type": attributes.get('instanceType'),
                            "vcpu": int(attributes.get('vcpu', '0')),
                            "memory": float(attributes.get('memory', '0 GiB').split()[0]),
                            "price": float(price)
                        }
                        prices[instance_type] = instance_data  # Store full data
                
                next_token = response.get('NextToken')
                if not next_token:
                    break
            
            # Update cache
            self.cache[cache_key] = prices
            self.last_updated[cache_key] = time.time()
            return prices
        
        except ClientError as e:
            print(f"Error fetching prices: {e}")
            return {}
    
    def get_ebs_pricing(self, region):
        # Simplified EBS pricing
        return {
            "gp2": 0.10,
            "gp3": {"gb": 0.08, "iops": 0.005, "throughput": 0.04},
            "io1": {"gb": 0.125, "iops": 0.065},
            "io2": {"gb": 0.125, "iops": 0.065},
            "aurora": 0.10
        }
    
    def get_backup_pricing(self, region):
        # Simplified backup pricing
        return 0.095  # per GB/month