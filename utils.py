# utils.py
# This file helps us load, understand and prepare our data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PART A: Column names for the NSL-KDD dataset
# (the dataset has no headers so we add them)
# ─────────────────────────────────────────────

COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'attack_type', 'difficulty'
]

# ─────────────────────────────────────────────
# PART B: Group all attack types into 1 label
# Normal = 0, Any attack = 1
# ─────────────────────────────────────────────

def label_attack(attack_name):
    """
    Convert attack name to binary label.
    normal  → 0  (safe)
    anything else → 1  (attack!)
    """
    if attack_name == 'normal':
        return 0
    else:
        return 1


# ─────────────────────────────────────────────
# PART C: Load the dataset
# ─────────────────────────────────────────────

def load_data(filepath='KDDTrain+.txt'):
    """
    Load the NSL-KDD dataset from file.
    
    What this function does step by step:
    1. Reads the text file into a table
    2. Adds column names (headers)
    3. Converts attack names to 0 or 1
    4. Returns the data table
    """
    print(f"📂 Loading data from: {filepath}")
    
    # Read the file into a pandas table (called a DataFrame)
    # sep=',' means each value is separated by a comma
    df = pd.read_csv(filepath, sep=',', names=COLUMN_NAMES)
    
    print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Convert attack names to 0 or 1
    df['label'] = df['attack_type'].apply(label_attack)
    
    # Count how many normal vs attack records we have
    normal_count = (df['label'] == 0).sum()
    attack_count = (df['label'] == 1).sum()
    
    print(f"✅ Normal traffic records: {normal_count}")
    print(f"✅ Attack traffic records: {attack_count}")
    print(f"✅ Attack percentage: {attack_count/len(df)*100:.1f}%")
    
    return df


# ─────────────────────────────────────────────
# PART D: Prepare features for the AI
# ─────────────────────────────────────────────

def prepare_features(df):
    """
    Get the data ready for the AI to learn from.
    
    Problems we fix here:
    1. Text columns — AI can't read text, only numbers
       So we convert: 'tcp' → 0, 'udp' → 1, 'icmp' → 2
    2. Big numbers — some features are huge (millions)
       So we scale everything between 0 and 1
    """
    print("\n🔧 Preparing features...")
    
    # --- Fix Problem 1: Convert text to numbers ---
    # These 3 columns contain text instead of numbers
    text_columns = ['protocol_type', 'service', 'flag']
    
    for col in text_columns:
        # Get all unique values in this column
        unique_values = df[col].unique()
        # Create a dictionary: e.g. {'tcp': 0, 'udp': 1, 'icmp': 2}
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        # Replace text with numbers
        df[col] = df[col].map(mapping)
        print(f"  ✅ Converted '{col}' text → numbers ({len(unique_values)} unique values)")
    
    # --- Choose which columns to use as features ---
    # We remove 'attack_type', 'difficulty', 'label' 
    # because those are answers, not inputs
    feature_columns = [col for col in df.columns 
                       if col not in ['attack_type', 'difficulty', 'label']]
    
    X = df[feature_columns].values   # Features (inputs)
    y = df['label'].values           # Labels (answers: 0 or 1)
    
    print(f"\n✅ Features shape: {X.shape}")
    print(f"   → {X.shape[0]} samples (rows)")
    print(f"   → {X.shape[1]} features (columns)")
    
    # --- Fix Problem 2: Scale all numbers between 0 and 1 ---
    # This is called "normalization" 
    # Think of it like converting all temperatures to the same scale
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # Avoid dividing by zero
    
    X_normalized = (X - X_min) / X_range
    
    print("✅ All features scaled between 0 and 1")
    
    return X_normalized, y, X_min, X_range


# ─────────────────────────────────────────────
# PART E: Split data into Train and Test sets
# ─────────────────────────────────────────────

def split_data(X, y, test_size=0.2):
    """
    Split data into training set and testing set.
    
    Why do we split?
    - Training set (80%): AI learns from this
    - Testing set  (20%): We test how well AI learned
                          (AI has NEVER seen this before)
    
    It's like studying from a textbook (train)
    and then taking an exam with new questions (test)
    """
    total = len(X)
    split_point = int(total * (1 - test_size))
    
    # Shuffle the data randomly first
    # (so training set isn't all normal and test isn't all attacks)
    indices = np.random.permutation(total)
    
    train_idx = indices[:split_point]
    test_idx  = indices[split_point:]
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test  = X[test_idx]
    y_test  = y[test_idx]
    
    print(f"\n✅ Data split complete:")
    print(f"   Training set: {len(X_train)} samples (80%)")
    print(f"   Testing set:  {len(X_test)} samples (20%)")
    
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────
# PART F: Draw a chart of the data
# ─────────────────────────────────────────────

def plot_data_distribution(y_train, y_test):
    """Draw a bar chart showing normal vs attack counts."""
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax, y, title in zip(axes, [y_train, y_test], 
                            ['Training Set', 'Testing Set']):
        normal = (y == 0).sum()
        attack = (y == 1).sum()
        
        bars = ax.bar(['Normal', 'Attack'], [normal, attack], 
                      color=['#2ecc71', '#e74c3c'], 
                      edgecolor='black', width=0.5)
        
        # Add numbers on top of bars
        for bar, count in zip(bars, [normal, attack]):
            ax.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 100,
                    str(count), ha='center', 
                    fontsize=12, fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Records')
        ax.set_ylim(0, max(normal, attack) * 1.15)
    
    plt.suptitle('Data Distribution: Normal vs Attack Traffic', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=150)
    plt.show()
    print("✅ Chart saved as data_distribution.png")