# main.py — Step 5: Full Training

from utils import load_data, prepare_features, split_data
from environment import CyberThreatEnv
from agent import DQNAgent
from train import train, evaluate, plot_training

print("=" * 60)
print("  ADAPTIVE CYBER THREAT DETECTION")
print("  Full Training Pipeline")
print("=" * 60)

# ── 1. Load & prepare data ──
print("\n📂 Loading data...")
df = load_data('KDDTrain+.txt')
X, y, X_min, X_range = prepare_features(df)
X_train, y_train, X_test, y_test = split_data(X, y)

# ── 2. Create environments ──
train_env = CyberThreatEnv(X_train, y_train)
test_env  = CyberThreatEnv(X_test,  y_test)

# ── 3. Create agent ──
agent = DQNAgent(
    state_size    = train_env.n_features,
    action_size   = train_env.action_space.n,
    learning_rate = 0.001,
    gamma         = 0.95,
    epsilon       = 1.0,
    epsilon_decay = 0.995,
    batch_size    = 64,
    memory_size   = 50000,
    target_update = 500
)

# ── 4. Train the agent ──
# episodes=5 first to see it working
# You can increase to 10-15 for better results
history = train(train_env, agent, episodes=5)

# ── 5. Evaluate on test data ──
metrics = evaluate(test_env, agent)

# ── 6. Save the trained model ──
agent.save('trained_model.npy')
print("\n💾 Model saved as trained_model.npy")

# ── 7. Plot training charts ──
plot_training(history)

print("\n" + "=" * 60)
print("  🎉 PROJECT TRAINING COMPLETE!")
print("=" * 60)