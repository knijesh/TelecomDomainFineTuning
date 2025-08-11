import pandas as pd
import random
from faker import Faker
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os

# Initialize Faker for realistic names/locations
fake = Faker()

# ========================
# 1. ENHANCED DATA CONFIGURATION
# ========================

# Fake telecom company name
COMPANY_NAME = "TeleMax Solutions"
PLANS = [
    {"name": "Basic 5G", "price": 40, "data": "5GB", "roaming": "None", "speed": "50Mbps", "type": "individual", "data_gb": 5},
    {"name": "Streamer Plus", "price": 60, "data": "15GB", "roaming": "$5/day", "speed": "100Mbps", "type": "individual", "data_gb": 15},
    {"name": "Global Traveler", "price": 75, "data": "Unlimited", "roaming": "10GB free", "speed": "200Mbps", "type": "individual", "data_gb": float('inf')},
    {"name": "Family Unlimited", "price": 120, "data": "Unlimited (4 lines)", "roaming": "5GB shared", "speed": "100Mbps", "type": "family", "data_gb": float('inf')},
    {"name": "Student Saver", "price": 35, "data": "8GB", "roaming": "$3/day", "speed": "75Mbps", "type": "individual", "data_gb": 8},
    {"name": "Business Pro", "price": 90, "data": "50GB", "roaming": "20GB included", "speed": "300Mbps", "type": "business", "data_gb": 50}
]

USAGE_PROFILES = [
    {"type": "light", "data_range": (1, 4), "call_range": (50, 150), "weight": 0.25},
    {"type": "medium", "data_range": (5, 12), "call_range": (150, 400), "weight": 0.45},
    {"type": "heavy", "data_range": (15, 35), "call_range": (400, 800), "weight": 0.25},
    {"type": "enterprise", "data_range": (30, 100), "call_range": (800, 1500), "weight": 0.05}
]

COMPLAINT_CATEGORIES = {
    "network": [
        "The signal drops constantly in downtown.",
        "I can't get any coverage at my home.",
        "Data speeds are incredibly slow during peak hours.",
        "Call quality is terrible - lots of static and dropped calls."
    ],
    "billing": [
        "Why am I being charged extra fees that weren't explained?",
        "My bill doubled this month with no explanation.",
        "I was promised a discount that never appeared on my bill.",
        "These roaming charges are outrageous for domestic travel."
    ],
    "customer_service": [
        "I've been waiting 2 weeks for a technician visit!",
        "Your customer service representatives are unhelpful and rude.",
        "I've called 5 times and get a different answer each time.",
        "It's impossible to reach a human representative."
    ],
    "technical": [
        "My phone keeps losing connection to your network.",
        "The mobile hotspot feature hasn't worked for days.",
        "I can't send or receive text messages reliably.",
        "Your network settings keep resetting on my device."
    ]
}

BUSINESS_NEEDS = [
    "team collaboration", "remote work", "client calls", "mobile office",
    "field operations", "international business", "data analytics", "video conferencing"
]

FAMILY_CONTEXTS = [
    "family of 4", "elderly parents", "college student", "teenagers",
    "mixed usage household", "multi-generational family", "frequent travelers"
]

# ========================
# 2. ENHANCED GENERATION FUNCTIONS
# ========================
def generate_usage_profile() -> Dict[str, Any]:
    """Generate a realistic usage profile with seasonal and demographic variations."""
    profile = random.choices(
        USAGE_PROFILES, 
        weights=[p['weight'] for p in USAGE_PROFILES]
    )[0]
    
    # Add seasonal variation (±20%)
    seasonal_factor = random.uniform(0.8, 1.2)
    
    return {
        "data_usage": max(1, int(random.randint(*profile['data_range']) * seasonal_factor)),
        "call_usage": max(50, int(random.randint(*profile['call_range']) * seasonal_factor)),
        "type": profile['type'],
        "seasonal_factor": seasonal_factor
    }

def find_best_plan(usage_gb: int, budget: int = None, plan_type: str = None) -> Dict:
    """Find the most suitable plan based on usage and constraints."""
    suitable_plans = []
    
    for plan in PLANS:
        # Filter by type if specified
        if plan_type and plan['type'] != plan_type:
            continue
            
        # Filter by budget if specified
        if budget and plan['price'] > budget:
            continue
            
        # Check if plan covers usage
        covers_usage = plan['data_gb'] == float('inf') or plan['data_gb'] >= usage_gb
        
        suitable_plans.append({
            **plan,
            'covers_usage': covers_usage,
            'value_score': plan['data_gb'] / plan['price'] if plan['data_gb'] != float('inf') else 10
        })
    
    if not suitable_plans:
        return random.choice(PLANS)
    
    # Prioritize plans that cover usage, then by value
    suitable_plans.sort(key=lambda x: (x['covers_usage'], x['value_score']), reverse=True)
    return suitable_plans[0]

def calculate_savings(new_plan: Dict, current_plan: Dict) -> Dict[str, float]:
    """Calculate comprehensive savings analysis."""
    monthly_savings = current_plan['price'] - new_plan['price']
    annual_savings = monthly_savings * 12
    
    return {
        "monthly": monthly_savings,
        "annual": annual_savings,
        "percentage": (monthly_savings / current_plan['price']) * 100 if current_plan['price'] > 0 else 0
    }

def generate_single_turn() -> Dict[str, str]:
    """Generate a single-turn plan recommendation dialogue."""
    current_plan = random.choice(PLANS)
    profile = generate_usage_profile()
    company_name = COMPANY_NAME
    
    # More varied question patterns
    question_templates = [
        f"I'm currently on {current_plan['name']} (${current_plan['price']}/month) with {company_name} but using {profile['data_usage']}GB monthly for {random.choice(['work', 'streaming', 'gaming', 'social media'])}. What's a better option?",
        f"My {current_plan['name']} plan from {company_name} isn't working for my {profile['data_usage']}GB usage. Need something more {random.choice(['affordable', 'flexible', 'comprehensive'])}.",
        f"Looking to switch from {current_plan['name']} at {company_name}. I need {profile['data_usage']}GB and better {random.choice(['coverage', 'speeds', 'value'])}.",
        f"Can you recommend an alternative to {current_plan['name']} from {company_name}? My usage is around {profile['data_usage']}GB and I want to {random.choice(['save money', 'get more data', 'improve service'])}."
    ]
    
    question = random.choice(question_templates)
    recommended_plan = find_best_plan(profile['data_usage'])
    savings = calculate_savings(recommended_plan, current_plan)
    
    # Enhanced answer with more details
    coverage_status = "Perfect fit" if recommended_plan['data_gb'] >= profile['data_usage'] else "May need monitoring"
    if recommended_plan['data_gb'] == float('inf'):
        coverage_status = "Unlimited - no worries!"
    
    answer = f"""Based on your {profile['data_usage']}GB usage pattern, I recommend the **{recommended_plan['name']} (${recommended_plan['price']}/month)** plan.

Key features:
• Data: {recommended_plan['data']} - {coverage_status}
• Speed: {recommended_plan['speed']}
• Roaming: {recommended_plan['roaming']}
• Monthly difference: ${abs(savings['monthly']):.0f} ({'save' if savings['monthly'] > 0 else 'additional cost'})
• Annual impact: ${abs(savings['annual']):.0f}

This plan suits {profile['type']} users and offers better value for your needs."""
    
    return {
        "input": question,
        "output": answer
    }

def generate_multi_turn() -> List[Dict[str, str]]:
    """Generate a multi-turn conversation as separate input/output pairs."""
    turns = []
    current_plan = random.choice(PLANS)
    profile = generate_usage_profile()
    company_name = COMPANY_NAME
    
    # Determine conversation context
    context = random.choice(["family", "business", "individual_upgrade"])
    
    # Initial question based on context
    if context == "family":
        user_context = random.choice(FAMILY_CONTEXTS)
        initial_question = f"I need a family plan for our {user_context}. Currently paying ${current_plan['price']} for {current_plan['name']} with {company_name}. What family options do you have?"
        recommended_plan = random.choice([p for p in PLANS if p['type'] == 'family'])
    elif context == "business":
        business_need = random.choice(BUSINESS_NEEDS)
        initial_question = f"Our business needs a plan for {business_need}. We're on {current_plan['name']} with {company_name} but need something more robust."
        recommended_plan = random.choice([p for p in PLANS if p['type'] in ['business', 'individual']])
    else:
        initial_question = f"I want to upgrade from {current_plan['name']} with {company_name}. Looking for better {random.choice(['data allowance', 'international options', 'speed', 'value'])}."
        recommended_plan = find_best_plan(profile['data_usage'] * 2)  # Assume they want more
    
    # First turn
    agent_response = f"Great choice! I recommend our **{recommended_plan['name']} (${recommended_plan['price']}/month)** which includes {recommended_plan['data']} data and {recommended_plan['speed']} speeds. {random.choice(['Would you like details about additional features?', 'Shall I explain the benefits?', 'Any specific requirements I should know about?'])}"
    
    turns.append({
        "input": initial_question,
        "output": agent_response
    })
    
    # Follow-up questions (1-2 additional turns)
    follow_up_options = [
        ("What's the contract length and cancellation policy?", "No annual contract required! You can switch anytime. We do offer a 10% discount if you choose a 12-month commitment."),
        ("How much for adding extra lines?", f"Additional lines are ${random.randint(15, 30)} each with the same features. Family plans get better per-line pricing."),
        ("Does this include mobile hotspot data?", f"Yes! Includes {random.choice(['5GB', '10GB', 'unlimited'])} mobile hotspot at full speed, then unlimited at reduced speed."),
        ("What about international roaming rates?", f"International roaming is {random.choice(['$5/day', '$10/day', 'included in select countries'])} with {random.choice(['1GB', '2GB', 'unlimited'])} daily allowance."),
        ("Are there any setup fees or hidden costs?", f"${random.choice([25, 35, 0])} activation fee {'waived this month' if random.choice([True, False]) else 'per line'}. No hidden charges!")
    ]
    
    num_followups = random.randint(1, 2)
    selected_followups = random.sample(follow_up_options, num_followups)
    
    for followup_question, followup_answer in selected_followups:
        turns.append({
            "input": followup_question,
            "output": followup_answer
        })
    
    return turns

def generate_complaint() -> Dict[str, str]:
    """Generate complaint handling scenarios."""
    category = random.choice(list(COMPLAINT_CATEGORIES.keys()))
    complaint = random.choice(COMPLAINT_CATEGORIES[category])
    company_name = random.choice(TELECOM_COMPANIES)
    
    # Add company context to complaint
    complaint_with_company = f"{complaint.replace('your', f'{company_name}s').replace('Your', f'{company_name}s')}"
    
    # Category-specific resolutions
    if category == "network":
        resolution = random.choice([
            f"I apologize for the coverage issues with {company_name}. I'm scheduling a network technician to check your area within 48 hours and will provide a service credit for the inconvenience.",
            f"Let me escalate this to our {company_name} network engineering team immediately. We'll also add a temporary signal booster to your account at no charge.",
            f"I understand your frustration with the {company_name} service quality. I'm applying a 25% credit to your next two bills while we resolve this network issue."
        ])
    elif category == "billing":
        resolution = random.choice([
            f"I sincerely apologize for the billing confusion with your {company_name} account. I've reviewed your account and I'm reversing the incorrect charges plus adding a $20 service credit.",
            f"You're absolutely right - this charge shouldn't be on your {company_name} bill. I'm processing a full refund and updating your account to prevent future billing errors.",
            f"Let me connect you with our {company_name} billing specialist who can explain these charges in detail and make any necessary corrections immediately."
        ])
    elif category == "customer_service":
        resolution = random.choice([
            f"I'm truly sorry for the poor service experience with {company_name}. I'm personally handling your case now and will ensure you have my direct number for any future needs.",
            f"This is unacceptable service from {company_name}, and I apologize. I'm escalating this to my supervisor and we'll have a senior technician contact you within 4 hours.",
            f"I understand your frustration with {company_name} completely. Let me arrange priority service for you and provide a significant account credit for this experience."
        ])
    else:  # technical
        resolution = random.choice([
            f"Let me troubleshoot this {company_name} technical issue right now. I'm also sending updated network settings to your device and scheduling a callback to ensure it's resolved.",
            f"I apologize for the technical problems with your {company_name} service. Our technical support team will call you within the hour, and I'm adding premium support to your account at no charge.",
            f"This sounds like a {company_name} network configuration issue. I'm pushing new settings to your device now and will monitor your connection for the next 24 hours."
        ])
    
    return {
        "input": complaint_with_company,
        "output": resolution
    }

# ========================
# 3. ENHANCED DATASET GENERATION
# ========================
def generate_dataset(num_samples: int = 1000, distribution: Dict[str, float] = None) -> List[Dict[str, str]]:
    """Generate dataset with configurable distribution and validation."""
    if distribution is None:
        distribution = {"single_turn": 0.5, "multi_turn": 0.35, "complaint": 0.15}
    
    dataset = []
    target_counts = {k: int(v * num_samples) for k, v in distribution.items()}
    
    print(f"Generating {num_samples} samples with distribution: {target_counts}")
    
    for dialog_type, count in target_counts.items():
        for i in range(count):
            if i % 100 == 0 and i > 0:
                print(f"  Generated {i}/{count} {dialog_type} samples...")
                
            if dialog_type == "single_turn":
                dataset.append(generate_single_turn())
            elif dialog_type == "multi_turn":
                # Multi-turn returns multiple input/output pairs
                multi_turn_data = generate_multi_turn()
                dataset.extend(multi_turn_data)
            elif dialog_type == "complaint":
                dataset.append(generate_complaint())
    
    # Fill remaining samples randomly if any
    current_count = len(dataset)
    remaining = num_samples - current_count
    
    if remaining > 0:
        for _ in range(remaining):
            choice = random.choices(
                list(distribution.keys()),
                weights=list(distribution.values())
            )[0]
            
            if choice == "single_turn":
                dataset.append(generate_single_turn())
            elif choice == "multi_turn":
                multi_turn_data = generate_multi_turn()
                # Only add first turn if we're close to target
                if len(dataset) + len(multi_turn_data) <= num_samples:
                    dataset.extend(multi_turn_data)
                else:
                    dataset.append(multi_turn_data[0])
            else:
                dataset.append(generate_complaint())
    
    # Trim to exact number if we exceeded
    dataset = dataset[:num_samples]
    
    # Shuffle the final dataset
    random.shuffle(dataset)
    return dataset

def validate_dataset(dataset: List[Dict]) -> Dict[str, Any]:
    """Validate dataset quality and provide statistics."""
    stats = {
        "total_samples": len(dataset),
        "type_distribution": {},
        "avg_dialog_length": 0,
        "plans_coverage": set(),
        "complaint_categories": {},
        "validation_errors": []
    }
    
    total_turns = 0
    
    for sample in dataset:
        # Type distribution
        sample_type = sample.get("type", "unknown")
        stats["type_distribution"][sample_type] = stats["type_distribution"].get(sample_type, 0) + 1
        
        # Dialog length
        dialog_length = len(sample.get("dialog", []))
        total_turns += dialog_length
        
        # Plans mentioned
        metadata = sample.get("metadata", {})
        if "recommended_plan" in metadata:
            stats["plans_coverage"].add(metadata["recommended_plan"])
        if "current_plan" in metadata:
            stats["plans_coverage"].add(metadata["current_plan"])
            
        # Complaint categories
        if sample_type == "complaint":
            category = metadata.get("complaint_category", "unknown")
            stats["complaint_categories"][category] = stats["complaint_categories"].get(category, 0) + 1
        
        # Validation checks
        if not sample.get("dialog"):
            stats["validation_errors"].append(f"Empty dialog in sample")
        
        for turn in sample.get("dialog", []):
            if not turn.get("speaker") or not turn.get("text"):
                stats["validation_errors"].append(f"Invalid turn structure")
    
    stats["avg_dialog_length"] = total_turns / len(dataset) if dataset else 0
    stats["plans_coverage"] = list(stats["plans_coverage"])
    
    return stats

# ========================
# 4. EXPORT & ANALYSIS FUNCTIONS
# ========================
def save_datasets(train_size: int = 800, test_size: int = 200, 
                 output_dir: str = ".", distribution: Dict[str, float] = None):
    """Save datasets in simple input/output JSONL format."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating datasets in {output_dir}/...")
    
    # Generate datasets
    train_data = generate_dataset(train_size, distribution)
    test_data = generate_dataset(test_size, distribution)
    
    # Save datasets in simple input/output format
    train_file = os.path.join(output_dir, 'telecom_train.jsonl')
    test_file = os.path.join(output_dir, 'telecom_test.jsonl')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n✅ Dataset Generation Complete!")
    print(f"📊 Training samples: {len(train_data)}")
    print(f"📊 Test samples: {len(test_data)}")
    print(f"📁 Files saved in: {output_dir}/")
    
    return train_data, test_data

# ========================
# 5. MAIN EXECUTION
# ========================
if __name__ == "__main__":
    # Custom distribution for different use cases
    balanced_dist = {"single_turn": 0.4, "multi_turn": 0.4, "complaint": 0.2}
    
    # Generate datasets
    train_data, test_data, stats = save_datasets(
        output_dir="telecom_dataset",
        train_size=1000,
        test_size=250,
        distribution=balanced_dist
    )
    
    # Google Colab integration
    try:
        from google.colab import files
        files.download('telecom_dataset/telecom_train.jsonl') 
        files.download('telecom_dataset/telecom_test.jsonl')
        files.download('telecom_dataset/dataset_stats.json')
        print("📥 Files downloaded to your Colab runtime")
    except ImportError:
        print("📁 Files saved locally (not in Colab environment)")

    # Display sample outputs
    print(f"\n" + "="*50)
    print("SAMPLE OUTPUTS")
    print("="*50)
    
    print("\n🗣️  Sample Single-Turn Dialogue:")
    single_sample = generate_single_turn()
    print(f'Input: "{single_sample["input"]}"')
    print(f'Output: "{single_sample["output"]}"')
    
    print(f"\n💬 Sample Multi-Turn Dialogue:")
    multi_samples = generate_multi_turn()
    for i, sample in enumerate(multi_samples, 1):
        print(f'Turn {i}:')
        print(f'  Input: "{sample["input"]}"')
        print(f'  Output: "{sample["output"]}"')
    
    print(f"\n⚠️  Sample Complaint Handling:")
    complaint_sample = generate_complaint()
    print(f'Input: "{complaint_sample["input"]}"')
    print(f'Output: "{complaint_sample["output"]}"')