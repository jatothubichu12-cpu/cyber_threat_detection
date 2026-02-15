# environment.py
# This is the "world" our AI lives in.
# It shows the AI one network record at a time
# and gives rewards for correct decisions.

import numpy as np


# ─────────────────────────────────────────────────────────
# HELPER CLASSES
# (These replace the gym library — no extra install needed)
# ─────────────────────────────────────────────────────────

class DiscreteSpace:
    """Represents a set of possible actions (0, 1, 2)."""
    def __init__(self, n):
        self.n = n  # number of actions

class BoxSpace:
    """Represents the observation (what AI sees)."""
    def __init__(self, shape):
        self.shape = shape  # shape of the input features


# ─────────────────────────────────────────────────────────
# MAIN ENVIRONMENT CLASS
# ─────────────────────────────────────────────────────────

class CyberThreatEnv:
    """
    The Cyber Threat Detection Environment.

    How it works:
    - Shows the AI one network traffic record at a time
    - AI makes a decision (Allow / Block / Flag)
    - Environment gives a reward based on correctness
    - Moves to the next record
    - Repeats until all records are seen (1 episode)

    Actions the AI can take:
        0 = Allow  (AI thinks this is normal traffic)
        1 = Block  (AI thinks this is an attack)
        2 = Flag   (AI is unsure — needs human review)

    Rewards the AI receives:
        +1.0  → Correctly blocked a real attack       ✅
        +1.0  → Correctly allowed normal traffic      ✅
        -1.0  → Missed an attack (very bad!)          ❌
        -0.5  → Blocked normal traffic (false alarm)  ⚠️
        +0.2  → Flagged something (neutral)           🔍
    """

    def __init__(self, data, labels):
        """
        Set up the environment with data.

        data   = the network traffic features (numbers)
        labels = the correct answers (0=normal, 1=attack)
        """
        # Store data as numpy arrays for fast processing
        self.data   = np.array(data,   dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int32)

        # Basic info about our data
        self.n_samples  = len(self.data)       # how many records total
        self.n_features = self.data.shape[1]   # how many features per record
        self.current_step = 0                   # which record we're on now

        # Define action space: AI can pick 0, 1, or 2
        self.action_space = DiscreteSpace(3)

        # Define observation space: AI sees 41 features
        self.observation_space = BoxSpace(shape=(self.n_features,))

        # Track statistics for this episode
        self.episode_stats = {
            'total_reward': 0,
            'correct':      0,
            'wrong':        0,
            'attacks_caught': 0,
            'attacks_missed': 0,
            'false_alarms':   0
        }

    # ─────────────────────────────────────────
    def reset(self):
        """
        Reset the environment to the beginning.
        Called at the start of every new episode.

        Returns: the first network traffic record
        """
        self.current_step = 0

        # Reset stats
        self.episode_stats = {
            'total_reward': 0,
            'correct':      0,
            'wrong':        0,
            'attacks_caught': 0,
            'attacks_missed': 0,
            'false_alarms':   0
        }

        # Return the first record for the AI to look at
        return self.data[self.current_step]

    # ─────────────────────────────────────────
    def step(self, action):
        """
        AI takes an action. Environment responds.

        This is the core function — called thousands of times during training.

        Input:  action (0, 1, or 2) — what the AI decided
        Output: (next_state, reward, done, info)
            next_state = the next traffic record to look at
            reward     = points for this decision
            done       = True if we've seen all records
            info       = extra details (for debugging)
        """
        # Get the correct answer for the current record
        true_label = self.labels[self.current_step]

        # Calculate reward based on AI's decision vs correct answer
        reward = self._get_reward(action, true_label)

        # Update statistics
        self.episode_stats['total_reward'] += reward
        if reward > 0:
            self.episode_stats['correct'] += 1
        else:
            self.episode_stats['wrong'] += 1

        if true_label == 1 and action == 1:
            self.episode_stats['attacks_caught'] += 1
        elif true_label == 1 and action == 0:
            self.episode_stats['attacks_missed'] += 1
        elif true_label == 0 and action == 1:
            self.episode_stats['false_alarms'] += 1

        # Move to the next record
        self.current_step += 1

        # Check if we've finished all records
        done = (self.current_step >= self.n_samples)

        # Get the next state (or zeros if we're done)
        if not done:
            next_state = self.data[self.current_step]
        else:
            next_state = np.zeros(self.n_features, dtype=np.float32)

        # Extra info for debugging
        info = {
            'true_label': true_label,
            'action':     action,
            'reward':     reward,
            'step':       self.current_step
        }

        return next_state, reward, done, info

    # ─────────────────────────────────────────
    def _get_reward(self, action, true_label):
        """
        Calculate reward based on action vs true label.

        Think of it like a teacher grading an answer:
        - Right answer on important question = +1
        - Wrong answer on important question = -1
        - Unnecessary wrong answer = -0.5
        """
        # AI said ALLOW (action=0)
        if action == 0:
            if true_label == 0:
                return +1.0   # ✅ Correct! Normal traffic allowed
            else:
                return -1.0   # ❌ Missed an attack! Very bad!

        # AI said BLOCK (action=1)
        elif action == 1:
            if true_label == 1:
                return +1.0   # ✅ Correct! Attack blocked!
            else:
                return -0.5   # ⚠️ False alarm — normal traffic blocked

        # AI said FLAG (action=2)
        elif action == 2:
            return +0.2       # 🔍 Neutral — human will review it

        return 0.0

    # ─────────────────────────────────────────
    def get_stats(self):
        """Return episode statistics in a readable format."""
        stats = self.episode_stats
        total  = stats['attacks_caught'] + stats['attacks_missed']
        detect = (stats['attacks_caught'] / total * 100) if total > 0 else 0

        return