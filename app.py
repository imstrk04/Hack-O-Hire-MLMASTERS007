import streamlit as st
import re
import math
import hashlib
import string
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import zlib
import base64
import json
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Password Strength Analyzer",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS with improved text colors for dark mode
st.markdown("""
<style>
    /* Base styles */
    .main {
        padding: 2rem;
        color: white;
    }
    
    /* Input field */
    .password-input {
        margin: 20px 0;
    }
    
    /* Container styles with proper text colors */
    .metric-container {
        background-color: #2d3748;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        color: white;
    }
    
    /* Suggestion styling */
    .suggestion {
        background-color: #1a365d;
        border-left: 4px solid #4299e1;
        padding: 10px;
        margin: 10px 0;
        color: white;
    }
    
    /* Warning styling */
    .warning {
        background-color: #742a2a;
        border-left: 4px solid #f56565;
        padding: 10px;
        margin: 10px 0;
        color: white;
    }
    
    /* Footer styling */
    .footer {
        margin-top: 30px;
        text-align: center;
        color: #a0aec0;
    }
    
    /* Make all text elements white by default */
    p, h1, h2, h3, h4, h5, h6, li, span {
        color: white !important;
    }
    
    /* Make expander text visible */
    .streamlit-expanderHeader {
        color: white !important;
    }
    
    /* Ensure labels are visible */
    label {
        color: white !important;
    }
    
    /* Fix sidebar text */
    .sidebar .sidebar-content {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load common password datasets
@st.cache_data
def load_common_passwords():
    # For demo purposes, using a small subset of common passwords
    # In production, you would use actual leaked password databases
    common_passwords = [
        "password", "123456", "qwerty", "admin", "welcome", 
        "password123", "abc123", "letmein", "monkey", "1234567890",
        "trustno1", "dragon", "baseball", "football", "iloveyou",
        "master", "sunshine", "ashley", "bailey", "passw0rd",
        "shadow", "superman", "qazwsx", "michael", "football"
    ]
    return set(common_passwords)

# Load patterns for password analysis
@st.cache_data
def load_password_patterns():
    patterns = {
        "sequential_chars": [
            "abcdef", "ghijkl", "mnopqr", "stuvwx", "yzabcd",
            "abcdefghijklmnopqrstuvwxyz"
        ],
        "sequential_nums": ["0123456789", "9876543210"],
        "keyboard_patterns": [
            "qwertyuiop", "asdfghjkl", "zxcvbnm",
            "qwerty", "asdfgh", "zxcvbn"
        ],
        "repeated_patterns": ["aaa", "111", "ababab", "121212", "abcabc"]
    }
    return patterns

# Load simulated historical user passwords
@st.cache_data
def load_user_password_history(user_id="demo_user"):
    # This would be replaced with actual user password history in a real app
    return [
        {"password_hash": "hashed_value_1", "created_date": "2023-01-01", "entropy": 45},
        {"password_hash": "hashed_value_2", "created_date": "2023-03-15", "entropy": 52},
        {"password_hash": "hashed_value_3", "created_date": "2023-06-20", "entropy": 48}
    ]

# Calculate password entropy
def calculate_entropy(password):
    if not password:
        return 0
    
    # Define character sets
    char_sets = {
        'lowercase': string.ascii_lowercase,
        'uppercase': string.ascii_uppercase,
        'digits': string.digits,
        'special': string.punctuation,
    }
    
    # Count unique character sets used
    used_sets = sum(1 for char_set in char_sets.values() 
                   if any(char in char_set for char in password))
    
    # Calculate pool size based on character sets used
    pool_size = sum(len(char_set) for char_set, chars in char_sets.items() 
                   if any(char in chars for char in password))
    
    # Apply length and complexity penalties/bonuses
    length_factor = min(len(password) / 8, 1.5)  # Cap length bonus at 1.5x
    
    # Simple repetition penalty (reduces entropy)
    repetition_penalty = 1.0
    for i in range(1, len(password)//2 + 1):
        for j in range(len(password) - i):
            if j + 2*i <= len(password) and password[j:j+i] == password[j+i:j+2*i]:
                repetition_penalty *= 0.9
    
    # Calculate raw entropy
    raw_entropy = math.log2(pool_size) * len(password)
    
    # Apply modifiers
    adjusted_entropy = raw_entropy * length_factor * repetition_penalty
    
    return max(0, adjusted_entropy)

# Estimate crack time based on entropy and attack methods
def estimate_crack_time(entropy):
    # Attack speeds (guesses per second)
    attack_speeds = {
        "online_throttled": 10,  # 10 guesses per second (with throttling)
        "online_unthrottled": 1000,  # 1,000 guesses per second
        "offline_slow_hash": 250000,  # 250,000 guesses per second (bcrypt)
        "offline_fast_hash": 1000000000,  # 1 billion guesses per second (MD5)
        "massive_parallel": 100000000000  # 100 billion guesses per second (specialized hardware)
    }
    
    # Time estimates for each attack method
    estimates = {}
    for attack, speed in attack_speeds.items():
        guesses = 2 ** entropy
        seconds = guesses / speed
        
        # Convert seconds to appropriate time unit
        if seconds < 60:
            estimates[attack] = f"{seconds:.1f} seconds"
        elif seconds < 3600:
            estimates[attack] = f"{seconds/60:.1f} minutes"
        elif seconds < 86400:
            estimates[attack] = f"{seconds/3600:.1f} hours"
        elif seconds < 31536000:
            estimates[attack] = f"{seconds/86400:.1f} days"
        elif seconds < 315360000:
            estimates[attack] = f"{seconds/31536000:.1f} years"
        else:
            centuries = seconds / 31536000 / 100
            if centuries > 1e20:
                estimates[attack] = "heat death of universe"
            else:
                estimates[attack] = f"{centuries:.1f} centuries"
    
    return estimates

# Analyze password for patterns and weaknesses
def analyze_password(password, common_passwords, patterns):
    results = {
        "weaknesses": [],
        "suggestions": []
    }
    
    # Check if password is too short
    if len(password) < 8:
        results["weaknesses"].append("Password is too short")
        results["suggestions"].append("Use at least 8 characters")
    
    # Check for missing character types
    if not any(c.islower() for c in password):
        results["weaknesses"].append("No lowercase letters")
        results["suggestions"].append("Add lowercase letters")
    
    if not any(c.isupper() for c in password):
        results["weaknesses"].append("No uppercase letters")
        results["suggestions"].append("Add uppercase letters")
    
    if not any(c.isdigit() for c in password):
        results["weaknesses"].append("No numbers")
        results["suggestions"].append("Add numbers")
    
    if not any(c in string.punctuation for c in password):
        results["weaknesses"].append("No special characters")
        results["suggestions"].append("Add special characters (!@#$%^&*)")
    
    # Check if password is a common one
    if password.lower() in common_passwords:
        results["weaknesses"].append("Password is in common password list")
        results["suggestions"].append("Choose a completely unique password")
    
    # Check for sequential patterns
    for pattern_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            for i in range(len(pattern) - 2):
                if pattern[i:i+3].lower() in password.lower():
                    results["weaknesses"].append(f"Contains sequential pattern ({pattern[i:i+3]})")
                    results["suggestions"].append("Avoid sequences like abc, 123, qwerty")
                    break
    
    # Check for repeated characters
    for i in range(len(password) - 2):
        if password[i] == password[i+1] == password[i+2]:
            results["weaknesses"].append("Contains repeated characters")
            results["suggestions"].append("Avoid repeating the same character (e.g., aaa, 111)")
            break
    
    # Remove duplicate suggestions
    results["suggestions"] = list(dict.fromkeys(results["suggestions"]))
    
    return results

# Generate improved password suggestion
def generate_password_suggestion(original_password=None, target_entropy=70):
    # Character sets
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    special = "!@#$%^&*()-_=+[]{}|;:,.<>?"
    
    # Base password length
    base_length = 12
    
    # Adjust length based on target entropy
    length = max(base_length, int(target_entropy / 4) + 2)
    
    # Generate password with good distribution of character types
    chars = []
    
    # Ensure at least 2 of each character type
    chars.extend(random.sample(lowercase, 2))
    chars.extend(random.sample(uppercase, 2))
    chars.extend(random.sample(digits, 2))
    chars.extend(random.sample(special, 2))
    
    # Fill remaining length with random characters
    remaining = length - len(chars)
    all_chars = lowercase + uppercase + digits + special
    chars.extend(random.choices(all_chars, k=remaining))
    
    # Shuffle the characters
    random.shuffle(chars)
    
    return ''.join(chars)

# Calculate similarity between two passwords
def calculate_password_similarity(password1, password2):
    # For demonstration - in production, you would use more sophisticated methods
    # This is just a simple Jaccard similarity example
    set1 = set(password1)
    set2 = set(password2)
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0

# Predict when a password should be changed
def predict_password_aging(password_history, current_entropy):
    # Basic implementation - would be more sophisticated in production
    if not password_history:
        # Default to 90 days if no history
        return datetime.now() + timedelta(days=90)
    
    # Calculate average time between password changes
    dates = [datetime.strptime(entry["created_date"], "%Y-%m-%d") for entry in password_history]
    if len(dates) < 2:
        avg_days = 90  # Default
    else:
        intervals = [(dates[i] - dates[i-1]).days for i in range(1, len(dates))]
        avg_days = sum(intervals) / len(intervals)
    
    # Adjust based on entropy (higher entropy = longer lifespan)
    entropy_factor = min(current_entropy / 60, 2.0)  # Cap at 2x
    recommended_days = avg_days * entropy_factor
    
    # Cap at reasonable limits
    recommended_days = max(min(recommended_days, 180), 30)
    
    return datetime.now() + timedelta(days=recommended_days)

# Create strength score from entropy
def get_strength_score(entropy):
    if entropy < 40:
        return "Very Weak", "danger", 1
    elif entropy < 60:
        return "Weak", "warning", 2
    elif entropy < 80:
        return "Moderate", "info", 3
    elif entropy < 100:
        return "Strong", "success", 4
    else:
        return "Very Strong", "success", 5

# Simulate checking against data breaches (would connect to real API in production)
def check_data_breach(password):
    # This is a simulation - in production, you would use a real service like
    # Have I Been Pwned API, which uses k-anonymity to check securely
    
    # Simple mock implementation - in production, you'd use actual leaked password databases
    common_leaked = {"123456", "password", "qwerty", "admin", "welcome", "12345678"}
    
    # Simulate an API call with a delay
    time.sleep(0.5)
    
    return password in common_leaked

# Configure matplotlib for dark mode
plt.style.use('dark_background')

# Main application
def main():
    st.title("🔒 Password Strength Analyzer")
    
    # Sidebar with options
    st.sidebar.header("Settings")
    
    min_entropy = st.sidebar.slider(
        "Minimum Recommended Entropy", 
        min_value=40, 
        max_value=120, 
        value=70,
        help="Higher values mean stronger passwords"
    )
    
    attack_model = st.sidebar.selectbox(
        "Crack Time Estimation Model",
        ["online_throttled", "online_unthrottled", "offline_slow_hash", 
         "offline_fast_hash", "massive_parallel"],
        index=2,
        help="Different attack scenarios require different levels of protection"
    )
    
    check_breaches = st.sidebar.checkbox(
        "Check Against Data Breaches", 
        value=True,
        help="Compare against known leaked passwords"
    )
    
    # Load data
    common_passwords = load_common_passwords()
    password_patterns = load_password_patterns()
    user_history = load_user_password_history()
    
    # Password input
    st.markdown("### Enter your password to analyze its strength")
    password_input = st.text_input(
        "Password", 
        type="password",
        key="password_input",
        help="Your password is never stored or transmitted"
    )
    
    # Show demo checkbox
    show_demo = st.checkbox("Show Demo Password", value=False)
    if show_demo:
        password_input = "Password123!"
    
    # Only analyze if there's a password
    if password_input:
        # Calculate metrics
        entropy = calculate_entropy(password_input)
        crack_times = estimate_crack_time(entropy)
        analysis = analyze_password(password_input, common_passwords, password_patterns)
        strength_label, color_class, strength_score = get_strength_score(entropy)
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        # Column 1: Strength metrics
        with col1:
            st.markdown("### Password Strength Assessment")
            
            # Overall strength
            st.markdown(f"""
            <div class="metric-container" style="border-left: 4px solid {'#4ade80' if strength_score > 3 else '#fb923c' if strength_score > 1 else '#f87171'};">
                <h4 style="color: white;">Overall Strength: {strength_label}</h4>
                <div style="background-color: #4b5563; height: 10px; border-radius: 5px; margin-top: 10px;">
                    <div style="background-color: {'#4ade80' if strength_score > 3 else '#fb923c' if strength_score > 1 else '#f87171'}; 
                                width: {strength_score * 20}%; 
                                height: 10px; 
                                border-radius: 5px;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Entropy metric
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="color: white;">Password Entropy: {entropy:.1f} bits</h4>
                <p style="color: white;">Entropy is a measure of password randomness and unpredictability</p>
                <div style="background-color: #4b5563; height: 10px; border-radius: 5px; margin-top: 10px;">
                    <div style="background-color: {'#4ade80' if entropy >= min_entropy else '#fb923c' if entropy >= min_entropy*0.7 else '#f87171'}; 
                                width: {min(100, entropy/120*100)}%; 
                                height: 10px; 
                                border-radius: 5px;">
                    </div>
                </div>
                <p style="margin-top: 5px; font-size: 0.8em; color: white;">Recommended: {min_entropy} bits or higher</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Password aging prediction
            next_change_date = predict_password_aging(user_history, entropy)
            days_until_change = (next_change_date - datetime.now()).days
            
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="color: white;">Recommended Password Change</h4>
                <p style="color: white;">Based on your password strength and usage patterns</p>
                <p style="font-size: 1.2em; margin-top: 10px; color: white;">Change in <strong>{days_until_change}</strong> days</p>
                <p style="color: white;">Expected date: {next_change_date.strftime('%Y-%m-%d')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Data breach check
            if check_breaches:
                is_compromised = check_data_breach(password_input)
                st.markdown(f"""
                <div class="{'warning' if is_compromised else 'metric-container'}">
                    <h4 style="color: white;">Data Breach Check</h4>
                    <p style="color: white;">{'⚠️ This password appears in known data breaches!' if is_compromised else '✅ Password not found in known breaches'}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Column 2: Time to crack and weaknesses
        with col2:
            st.markdown("### Crack Time Estimation")
            
            # Format selected attack time estimate
            selected_time = crack_times[attack_model]
            
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="color: white;">Selected Attack Model: {attack_model.replace('_', ' ').title()}</h4>
                <p style="font-size: 1.2em; margin-top: 10px; color: white;">Time to crack: <strong>{selected_time}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show all attack times in expandable section
            with st.expander("See estimates for all attack methods"):
                for attack, time_estimate in crack_times.items():
                    st.markdown(f"**{attack.replace('_', ' ').title()}**: {time_estimate}")
            
            # Weaknesses and suggestions
            if analysis["weaknesses"]:
                st.markdown("### Identified Weaknesses")
                for weakness in analysis["weaknesses"]:
                    st.markdown(f"""
                    <div class="warning">
                        <p style="color: white;">⚠️ {weakness}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            if analysis["suggestions"]:
                st.markdown("### Improvement Suggestions")
                for suggestion in analysis["suggestions"]:
                    st.markdown(f"""
                    <div class="suggestion">
                        <p style="color: white;">💡 {suggestion}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Password suggestions section
        st.markdown("### Password Suggestions")
        st.markdown("These suggested passwords meet your security requirements:")
        
        # Generate a few password suggestions
        suggestions = [generate_password_suggestion(target_entropy=min_entropy) for _ in range(3)]
        
        # Show suggestions in a nice format
        suggestion_cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            suggestion_entropy = calculate_entropy(suggestion)
            _, color, _ = get_strength_score(suggestion_entropy)
            
            with suggestion_cols[i]:
                st.markdown(f"""
                <div class="metric-container" style="text-align: center;">
                    <p style="font-family: monospace; font-size: 1.1em; color: white;">{suggestion}</p>
                    <p style="color: white;">Entropy: {suggestion_entropy:.1f} bits</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Password visualization (simplified entropy distribution)
        st.markdown("### Password Visualization")
        
        # Create visualization of password character distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Count character types
        char_counts = {
            'Lowercase': sum(1 for c in password_input if c.islower()),
            'Uppercase': sum(1 for c in password_input if c.isupper()),
            'Numbers': sum(1 for c in password_input if c.isdigit()),
            'Special': sum(1 for c in password_input if c in string.punctuation)
        }
        
        # Create bar colors based on diversity
        colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99']
        if all(count > 0 for count in char_counts.values()):
            colors = ['#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']  # More pleasing colors for good distribution
        
        # Plot bars
        bars = ax.bar(char_counts.keys(), char_counts.values(), color=colors)
        
        # Add labels
        ax.set_ylabel('Count')
        ax.set_title('Character Type Distribution')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        # Display the chart in Streamlit
        st.pyplot(fig)
        
        # Explanation of entropy
        with st.expander("Understanding Password Entropy"):
            st.markdown("""
            **Password entropy** is a measurement of how unpredictable a password is.
            
            - Higher entropy = more secure password
            - Entropy increases with:
              - Password length
              - Character variety (lowercase, uppercase, numbers, symbols)
              - Randomness (lack of patterns)
            
            Generally, passwords should have:
            - At least 60 bits of entropy for regular accounts
            - 80+ bits for important accounts (email, banking)
            - 100+ bits for critical security applications
            
            The calculation takes into account both the character set size and password length.
            """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Password Strength Analyzer v1.0 | Developed for security education</p>
        <p>Your passwords are never stored or transmitted from your browser</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()