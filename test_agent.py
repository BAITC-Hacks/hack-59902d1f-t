#!/usr/bin/env python3
"""
Simple local test of the agent
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from agent import Agent


class SimpleTestEnv:
    """Minimal test environment for agent validation"""
    
    def __init__(self, n_customers=1000):
        self.n_customers = n_customers
        
        # Create synthetic customer profile
        np.random.seed(42)
        self.customer_profile = pd.DataFrame({
            'customer_id': range(n_customers),
            'current_tariff': np.random.choice(
                [f'tariff_{i}' for i in range(1, 11)], n_customers
            ),
            'arpu_segment': np.random.choice(['LOW', 'MID', 'HIGH'], n_customers, p=[0.5, 0.35, 0.15]),
            'data_segment': np.random.choice(['NON_USER', 'LITE', 'HEAVY'], n_customers, p=[0.3, 0.4, 0.3]),
            'call_segment': np.random.choice(['LOW', 'MEDIUM', 'HIGH'], n_customers),
            'predicted_arpu': np.random.lognormal(7, 1.5, n_customers),  # Log-normal distribution
        })
        
        self.tariffs = [f'tariff_{i}' for i in range(1, 22)]
        self.channels = ['push', 'sms', 'digital_ads', 'call']
        
        self.remaining_budget = 100000
        self.remaining_contacts = 15000
        self.pilots_left = 20
        
        self.pilot_count = 0
        
    def run_pilot(self, target_tariff: str, channel: str, n_customers: int,
                  filter_arpu_segment=None, filter_data_segment=None,
                  filter_current_tariff=None) -> Dict:
        """
        Simulate a pilot campaign
        """
        self.pilot_count += 1
        
        # Generate synthetic results
        # In reality, these depend on the actual segment
        np.random.seed(100 + self.pilot_count)  # Different seed per pilot
        
        base_effect = {
            'tariff_18': 180,
            'tariff_19': 200,
            'tariff_20': 220,
        }
        
        mean_change = base_effect.get(target_tariff, 100) + np.random.normal(0, 50)
        std_change = max(10, np.random.normal(80, 30))
        conversion_rate = np.random.uniform(0.2, 0.4)
        
        channel_cost = {
            'push': 0,
            'sms': 4,
            'digital_ads': 22,
            'call': 160
        }
        
        cost = n_customers * channel_cost[channel]
        
        # Update budget and contacts
        self.remaining_budget -= cost
        self.remaining_contacts -= n_customers
        self.pilots_left -= 1
        
        result = {
            'mean_arpu_change': mean_change,
            'std_arpu_change': std_change,
            'conversion_rate': conversion_rate,
            'n_customers': n_customers,
            'cost': cost
        }
        
        print(f"  [Pilot {self.pilot_count}] {target_tariff} via {channel}: "
              f"mean_arpu={mean_change:.1f}, conv={conversion_rate:.2f}, "
              f"cost={cost} у.е.")
        
        return result


def test_agent():
    """Test agent on synthetic data"""
    
    print("=" * 60)
    print("Testing Tariff Marketing Agent")
    print("=" * 60)
    
    # Create test environment
    env = SimpleTestEnv(n_customers=2000)
    
    print(f"\n[ENV] Setup:")
    print(f"  - Customers: {env.n_customers}")
    print(f"  - Tariffs: {len(env.tariffs)}")
    print(f"  - Channels: {env.channels}")
    print(f"  - Budget: {env.remaining_budget:,} у.е.")
    print(f"  - Contact limit: {env.remaining_contacts:,}")
    print(f"  - Pilots available: {env.pilots_left}")
    
    # Create and run agent
    print(f"\n[AGENT] Running...")
    agent = Agent()
    
    try:
        campaigns = agent.act(env)
        
        print(f"\n[RESULT] Generated {len(campaigns)} campaigns:")
        print("-" * 60)
        
        if not campaigns:
            print("  No campaigns generated (might be due to constraints)")
            return False
        
        for i, campaign in enumerate(campaigns):
            print(f"\n  Campaign {i+1}:")
            for key, value in campaign.items():
                print(f"    {key}: {value}")
        
        print(f"\n[SUMMARY]:")
        print(f"  - Campaigns: {len(campaigns)}")
        print(f"  - Pilots executed: {env.pilot_count}")
        print(f"  - Budget remaining: {env.remaining_budget:,} у.е.")
        print(f"  - Contacts remaining: {env.remaining_contacts:,}")
        
        # Validation
        print(f"\n[VALIDATION]:")
        
        # Check campaign count
        if len(campaigns) > 10:
            print("  ❌ Too many campaigns (max 10)")
            return False
        else:
            print(f"  ✓ Campaign count OK ({len(campaigns)} <= 10)")
        
        # Check all campaigns have required fields
        required_fields = ['target_tariff', 'channel']
        for i, campaign in enumerate(campaigns):
            for field in required_fields:
                if field not in campaign:
                    print(f"  ❌ Campaign {i} missing field: {field}")
                    return False
        print(f"  ✓ All campaigns have required fields")
        
        # Check tariffs exist
        for i, campaign in enumerate(campaigns):
            if campaign['target_tariff'] not in env.tariffs:
                print(f"  ❌ Campaign {i} has invalid tariff: {campaign['target_tariff']}")
                return False
        print(f"  ✓ All tariffs valid")
        
        # Check channels exist
        for i, campaign in enumerate(campaigns):
            if campaign['channel'] not in env.channels:
                print(f"  ❌ Campaign {i} has invalid channel: {campaign['channel']}")
                return False
        print(f"  ✓ All channels valid")
        
        # Check budget not exceeded
        if env.remaining_budget < 0:
            print(f"  ❌ Budget exceeded by {-env.remaining_budget:,} у.е.")
            return False
        else:
            print(f"  ✓ Budget OK")
        
        # Check contacts not exceeded
        if env.remaining_contacts < 0:
            print(f"  ❌ Contacts exceeded by {-env.remaining_contacts:,}")
            return False
        else:
            print(f"  ✓ Contacts OK")
        
        print(f"\n[STATUS] ✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_agent()
    exit(0 if success else 1)
