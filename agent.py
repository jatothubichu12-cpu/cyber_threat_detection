# agent.py
# This is the AI BRAIN — the DQN Agent
# It learns from experience to detect cyber attacks

import numpy as np
import random
from collections import deque


# ═══════════════════════════════════════════════════════
# PART 1: THE NEURAL NETWORK
# ═══════════════════════════════════════════════════════
# A neural network is like a brain made of layers.
# Each layer transforms the input into something more useful.
#
# Our network:
#   Input Layer  → 41 features (what AI sees)
#   Hidden Layer → 128 neurons (AI thinks here)
#   Output Layer → 3 values (scores for Allow/Block/Flag)
#
# The output scores are called Q-values.
# Higher Q-value = AI thinks that action is better.
# ═══════════════════════════════════════════════════════

class NeuralNetwork:
    """
    A simple 2-layer neural network built from scratch.
    No TensorFlow or PyTorch needed — just numpy!
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        """
        Set up the network with random starting weights.

        Weights are like the "strength" of connections between neurons.
        They start random and improve during training.
        """
        self.lr = learning_rate

        # ── Layer 1 weights: input → hidden ──
        # Xavier initialization: keeps values from getting too big/small
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))  # bias = starting point for each neuron

        # ── Layer 2 weights: hidden → output ──
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, x):
        """
        Pass input through the network to get Q-values.

        Think of it like water flowing through pipes:
        Input → Layer1 (with ReLU) → Layer2 → Output Q-values

        ReLU activation: turns negative numbers to 0
        (like a valve that only lets positive signals through)
        """
        # Make sure input is 2D: shape (1, 41)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Layer 1: multiply by weights, add bias, apply ReLU
        self.input = x
        self.z1 = x @ self.W1 + self.b1          # raw calculation
        self.a1 = np.maximum(0, self.z1)           # ReLU: max(0, x)

        # Layer 2: multiply by weights, add bias (no activation)
        self.z2 = self.a1 @ self.W2 + self.b2     # Q-values output

        return self.z2  # shape: (1, 3) — scores for Allow, Block, Flag

    def train_step(self, x, action, target_q_value):
        """
        Update weights to improve prediction for one experience.

        This is called 'backpropagation' — we:
        1. Calculate how wrong we were (the error)
        2. Figure out which weights caused the error
        3. Adjust those weights to be less wrong next time
        """
        # Forward pass to get current prediction
        q_values = self.forward(x)

        # Calculate error only for the action we took
        error = target_q_value - q_values[0][action]

        # ── Backpropagation ──
        # Gradient for output layer
        dL_dz2 = np.zeros_like(q_values)
        dL_dz2[0][action] = -2 * error

        # Gradient for Layer 2 weights
        dW2 = self.a1.T @ dL_dz2
        db2 = dL_dz2

        # Gradient flowing back to Layer 1
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * (self.z1 > 0)  # ReLU gradient

        # Gradient for Layer 1 weights
        dW1 = self.input.T @ dL_dz1
        db1 = dL_dz1

        # ── Update weights (move in direction that reduces error) ──
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return error ** 2  # return loss (how wrong we were)

    def get_weights(self):
        """Return a copy of all weights."""
        return (self.W1.copy(), self.b1.copy(),
                self.W2.copy(), self.b2.copy())

    def set_weights(self, weights):
        """Set weights from a copy (used for target network)."""
        self.W1, self.b1, self.W2, self.b2 = [w.copy() for w in weights]


# ═══════════════════════════════════════════════════════
# PART 2: THE MEMORY (REPLAY BUFFER)
# ═══════════════════════════════════════════════════════
# The AI stores past experiences here.
# Instead of learning only from the latest event,
# it randomly picks old memories to learn from too.
#
# Why? Because if the AI only learned from recent events,
# it would forget what it learned before!
# It's like studying random pages of your notes
# instead of only the last page.
# ═══════════════════════════════════════════════════════

class ReplayMemory:
    """Stores past (state, action, reward, next_state, done) experiences."""

    def __init__(self, max_size=50000):
        # deque = a list that automatically removes oldest items when full
        self.memory = deque(maxlen=max_size)

    def store(self, state, action, reward, next_state, done):
        """Save one experience."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Randomly pick a batch of past experiences to learn from."""
        return random.sample(self.memory, batch_size)

    def size(self):
        return len(self.memory)


# ═══════════════════════════════════════════════════════
# PART 3: THE DQN AGENT (puts it all together)
# ═══════════════════════════════════════════════════════

class DQNAgent:
    """
    The complete DQN Agent.

    It has:
    - A main network  (learns and updates every step)
    - A target network (stable copy, updated slowly)
    - A memory        (stores past experiences)
    - Epsilon         (controls explore vs exploit)

    Why TWO networks?
    If we used the same network for both learning AND
    as the target, the target keeps changing → unstable!
    It's like trying to hit a moving target.
    The target network stays frozen for a while → stable learning.
    """

    def __init__(self,
                 state_size,         # number of features (41)
                 action_size,        # number of actions (3)
                 learning_rate=0.001,
                 gamma=0.95,         # discount factor (how much future rewards matter)
                 epsilon=1.0,        # starting exploration rate (100% random)
                 epsilon_min=0.01,   # minimum exploration rate (1% random)
                 epsilon_decay=0.995,# how fast to reduce randomness
                 batch_size=64,      # how many memories to learn from at once
                 memory_size=50000,  # max memories to store
                 target_update=500): # how often to sync target network

        self.state_size   = state_size
        self.action_size  = action_size
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay= epsilon_decay
        self.batch_size   = batch_size
        self.target_update= target_update
        self.train_count  = 0  # counts how many times we've trained

        # Create main network and target network
        hidden_size = 128
        self.main_network   = NeuralNetwork(state_size, hidden_size,
                                            action_size, learning_rate)
        self.target_network = NeuralNetwork(state_size, hidden_size,
                                            action_size, learning_rate)

        # Sync them at the start — both start identical
        self.target_network.set_weights(self.main_network.get_weights())

        # Create memory
        self.memory = ReplayMemory(memory_size)

        # Track training progress
        self.losses  = []
        self.epsilons = []

    # ─────────────────────────────────────────
    def act(self, state):
        """
        Choose an action given the current state.

        Uses Epsilon-Greedy strategy:
        - With probability epsilon  → pick RANDOM action (explore)
        - With probability 1-epsilon → pick BEST action  (exploit)

        At the start epsilon=1.0 (all random = lots of exploring)
        Over time epsilon shrinks → AI trusts itself more
        """
        if np.random.rand() < self.epsilon:
            # Explore: pick a random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit: ask the network which action looks best
            q_values = self.main_network.forward(state)
            return int(np.argmax(q_values[0]))  # pick highest Q-value

    # ─────────────────────────────────────────
    def remember(self, state, action, reward, next_state, done):
        """Store an experience in memory."""
        self.memory.store(state, action, reward, next_state, done)

    # ─────────────────────────────────────────
    def learn(self):
        """
        Learn from a random batch of past experiences.

        This is where the real learning happens!

        The Bellman Equation (the key formula):
        Target Q = reward + gamma × max(future Q-values)

        In simple words:
        'The value of this action = immediate reward
         + discounted value of the best future action'

        Like deciding whether to study now:
        Value = grade boost now + 0.95 × best future outcome
        """
        # Only start learning once we have enough memories
        if self.memory.size() < self.batch_size:
            return 0.0

        # Pick random batch of memories
        batch = self.memory.sample(self.batch_size)
        total_loss = 0.0

        for state, action, reward, next_state, done in batch:

            if done:
                # No future — target is just the immediate reward
                target_q = reward
            else:
                # Use TARGET network to estimate future rewards (stable!)
                future_q = self.target_network.forward(next_state)
                target_q = reward + self.gamma * np.max(future_q[0])

            # Train main network to predict this target
            loss = self.main_network.train_step(state, action, target_q)
            total_loss += loss

        # Reduce epsilon (AI explores less as it gets smarter)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Every N steps: copy main network weights to target network
        self.train_count += 1
        if self.train_count % self.target_update == 0:
            self.target_network.set_weights(self.main_network.get_weights())

        avg_loss = total_loss / self.batch_size
        self.losses.append(avg_loss)
        self.epsilons.append(self.epsilon)
        return avg_loss

    # ─────────────────────────────────────────
    def save(self, filepath):
        """Save the trained model weights to a file."""
        weights = {
            'W1': self.main_network.W1,
            'b1': self.main_network.b1,
            'W2': self.main_network.W2,
            'b2': self.main_network.b2,
            'epsilon': np.array([self.epsilon])
        }
        np.save(filepath, weights, allow_pickle=True)
        print(f"💾 Model saved to {filepath}")

    def load(self, filepath):
        """Load previously trained model weights from a file."""
        weights = np.load(filepath, allow_pickle=True).item()
        w = (weights['W1'], weights['b1'],
             weights['W2'], weights['b2'])
        self.main_network.set_weights(w)
        self.target_network.set_weights(w)
        self.epsilon = float(weights['epsilon'][0])
        print(f"📂 Model loaded from {filepath}")