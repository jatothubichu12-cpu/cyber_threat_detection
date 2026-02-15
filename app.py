# app.py
# This is the web server that connects
# your trained AI to the frontend dashboard

from flask import Flask, render_template, jsonify
from agent import DQNAgent
from utils import load_data, prepare_features, split_data
from environment import CyberThreatEnv
import numpy as np
import threading
import time
import random

app = Flask(__name__)

# ── Global state (shared between AI and dashboard) ──
state = {
    'running':        False,
    'total':          0,
    'attacks_caught': 0,
    'attacks_missed': 0,
    'false_alarms':   0,
    'normal_allowed': 0,
    'recent_events':  [],   # last 10 events for live feed
    'detection_rate': 0.0,
    'accuracy':       0.0,
}

# ── Load trained AI ──
print("🧠 Loading trained AI model...")
df = load_data('KDDTrain+.txt')
X, y, X_min, X_range = prepare_features(df)
X_train, y_train, X_test, y_test = split_data(X, y)

env   = CyberThreatEnv(X_test, y_test)
agent = DQNAgent(
    state_size  = env.n_features,
    action_size = env.action_space.n
)
agent.load('trained_model.npy')
agent.epsilon = 0.0  # No exploration — use learned knowledge only
print("✅ AI model loaded and ready!")


# ── Background thread: runs AI detection ──
def run_detection():
    """
    Runs the AI in the background, updating
    the global state so the dashboard can display it.
    """
    global state
    obs = env.reset()
    state['running'] = True

    action_names = {0: 'Allowed', 1: 'Blocked', 2: 'Flagged'}
    label_names  = {0: 'Normal',  1: 'Attack'}

    while True:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)

        true_label = info['true_label']
        state['total'] += 1

        # Update counters
        if true_label == 1 and action == 1:
            state['attacks_caught'] += 1
            status = 'success'
        elif true_label == 1 and action == 0:
            state['attacks_missed'] += 1
            status = 'danger'
        elif true_label == 0 and action == 0:
            state['normal_allowed'] += 1
            status = 'success'
        else:
            state['false_alarms'] += 1
            status = 'warning'

        # Calculate rates
        attacks = state['attacks_caught'] + state['attacks_missed']
        correct = state['attacks_caught'] + state['normal_allowed']

        state['detection_rate'] = round(
            state['attacks_caught'] / attacks * 100
            if attacks > 0 else 0, 1)

        state['accuracy'] = round(
            correct / state['total'] * 100
            if state['total'] > 0 else 0, 1)

        # Add to live event feed (keep last 8)
        event = {
            'id':     state['total'],
            'label':  label_names[true_label],
            'action': action_names[action],
            'reward': round(reward, 1),
            'status': status
        }
        state['recent_events'].insert(0, event)
        state['recent_events'] = state['recent_events'][:8]

        obs = next_obs

        # Reset when done
        if done:
            obs = env.reset()

        # Small delay so dashboard can update smoothly
        time.sleep(0.05)


# ── Start AI in background when server starts ──
thread = threading.Thread(target=run_detection, daemon=True)
thread.start()


# ── Web Routes ──
@app.route('/')
def dashboard():
    """Serve the main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/stats')
def get_stats():
    """API endpoint — returns current stats as JSON."""
    return jsonify(state)


if __name__ == '__main__':
    print("\n🌐 Dashboard running at: http://127.0.0.1:5000")
    print("   Open this link in your browser!")
    app.run(debug=False)