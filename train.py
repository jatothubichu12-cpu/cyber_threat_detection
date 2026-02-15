# train.py
# This file runs the full training loop
# and shows progress as the AI gets smarter

import numpy as np
import matplotlib.pyplot as plt
import time


# ═══════════════════════════════════════════════════════
# PART 1: THE TRAINING LOOP
# ═══════════════════════════════════════════════════════

def train(env, agent, episodes=10):
    """
    Train the AI agent for a number of episodes.

    What is an episode?
    → One full pass through ALL training data.
    → Like reading the entire textbook once.
    → After each episode the AI has learned more.

    We run multiple episodes so the AI sees the
    same data multiple times and keeps improving.
    """

    print("\n" + "=" * 60)
    print("   TRAINING STARTED")
    print("=" * 60)
    print(f"   Episodes     : {episodes}")
    print(f"   Training data: {env.n_samples} records")
    print(f"   Features     : {env.n_features}")
    print("=" * 60)

    # Store results from each episode for plotting later
    history = {
        'episode':        [],
        'total_reward':   [],
        'detection_rate': [],
        'false_alarms':   [],
        'accuracy':       [],
        'epsilon':        [],
        'avg_loss':       []
    }

    training_start = time.time()

    for episode in range(1, episodes + 1):

        episode_start = time.time()

        # Reset environment to beginning
        state = env.reset()

        # Counters for this episode
        total_reward    = 0
        attacks_caught  = 0
        attacks_missed  = 0
        false_alarms    = 0
        correct         = 0
        total_steps     = 0

        # ── Run through ALL records in the dataset ──
        while True:

            # 1. Agent decides what to do
            action = agent.act(state)

            # 2. Environment responds
            next_state, reward, done, info = env.step(action)

            # 3. Agent stores this experience in memory
            agent.remember(state, action, reward, next_state, done)

            # 4. Agent learns from a random batch of memories
            agent.learn()

            # 5. Update counters
            total_reward += reward
            total_steps  += 1
            true_label    = info['true_label']

            if true_label == 1 and action == 1:
                attacks_caught += 1   # ✅ Attack correctly blocked
                correct        += 1
            elif true_label == 1 and action == 0:
                attacks_missed += 1   # ❌ Attack missed!
            elif true_label == 0 and action == 0:
                correct        += 1   # ✅ Normal correctly allowed
            elif true_label == 0 and action == 1:
                false_alarms   += 1   # ⚠️ False alarm

            # 6. Move to next state
            state = next_state

            # 7. Stop when all records are seen
            if done:
                break

        # ── Calculate episode metrics ──
        total_attacks  = attacks_caught + attacks_missed
        detection_rate = (attacks_caught / total_attacks * 100) if total_attacks > 0 else 0
        accuracy       = (correct / total_steps * 100) if total_steps > 0 else 0
        avg_loss       = np.mean(agent.losses[-1000:]) if agent.losses else 0
        episode_time   = time.time() - episode_start

        # ── Save to history ──
        history['episode'].append(episode)
        history['total_reward'].append(total_reward)
        history['detection_rate'].append(detection_rate)
        history['false_alarms'].append(false_alarms)
        history['accuracy'].append(accuracy)
        history['epsilon'].append(agent.epsilon)
        history['avg_loss'].append(avg_loss)

        # ── Print episode summary ──
        print(f"\n{'─'*60}")
        print(f"  Episode {episode}/{episodes}  "
              f"[{episode_time:.1f}s]")
        print(f"{'─'*60}")
        print(f"  🎯 Detection Rate : {detection_rate:.1f}%  "
              f"({attacks_caught} caught / {total_attacks} total attacks)")
        print(f"  ✅ Accuracy       : {accuracy:.1f}%")
        print(f"  ⚠️  False Alarms  : {false_alarms}")
        print(f"  🏆 Total Reward   : {total_reward:.0f}")
        print(f"  📉 Avg Loss       : {avg_loss:.4f}")
        print(f"  🎲 Epsilon        : {agent.epsilon:.4f}  "
              f"(exploration rate)")

        # Show improvement tip
        if episode == 1:
            print(f"\n  💡 Episode 1 done! AI is just starting to learn.")
        elif detection_rate >= 95:
            print(f"\n  🔥 Excellent! Detection rate above 95%!")
        elif detection_rate >= 85:
            print(f"\n  📈 Good progress! Getting better each episode.")

    # ── Training complete ──
    total_time = time.time() - training_start
    print(f"\n{'=' * 60}")
    print(f"  ✅ TRAINING COMPLETE!")
    print(f"  Total time    : {total_time:.1f} seconds")
    print(f"  Final accuracy: {history['accuracy'][-1]:.1f}%")
    print(f"  Final detect  : {history['detection_rate'][-1]:.1f}%")
    print(f"{'=' * 60}")

    return history


# ═══════════════════════════════════════════════════════
# PART 2: EVALUATE THE TRAINED AGENT
# ═══════════════════════════════════════════════════════

def evaluate(env, agent):
    """
    Test the trained agent on data it has NEVER seen before.

    This is like the final exam after all the studying.
    We turn off all exploration (epsilon = 0) so the
    agent uses only what it has learned.
    """
    print("\n" + "=" * 60)
    print("  EVALUATING ON TEST DATA (never seen before)")
    print("=" * 60)

    # Turn off exploration — use only learned knowledge
    old_epsilon    = agent.epsilon
    agent.epsilon  = 0.0

    state = env.reset()

    TP = FP = TN = FN = 0  # True/False Positives/Negatives

    while True:
        action                  = agent.act(state)
        next_state, _, done, info = env.step(action)
        true_label              = info['true_label']

        # Count results
        if   true_label == 1 and action == 1: TP += 1  # Attack caught ✅
        elif true_label == 0 and action == 1: FP += 1  # False alarm ⚠️
        elif true_label == 0 and action == 0: TN += 1  # Normal allowed ✅
        elif true_label == 1 and action == 0: FN += 1  # Attack missed ❌

        state = next_state
        if done:
            break

    # Restore epsilon
    agent.epsilon = old_epsilon

    # Calculate all metrics
    total     = TP + FP + TN + FN
    accuracy  = (TP + TN) / total * 100
    precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    print(f"\n  Results on {total} test records:")
    print(f"  {'─'*40}")
    print(f"  ✅ True Positives  (attacks caught) : {TP:6d}")
    print(f"  ✅ True Negatives  (normal allowed) : {TN:6d}")
    print(f"  ⚠️  False Positives (false alarms)  : {FP:6d}")
    print(f"  ❌ False Negatives (attacks missed) : {FN:6d}")
    print(f"  {'─'*40}")
    print(f"  🎯 Accuracy  : {accuracy:.2f}%")
    print(f"  🔍 Precision : {precision:.2f}%")
    print(f"  📡 Recall    : {recall:.2f}%")
    print(f"  ⭐ F1-Score  : {f1:.2f}%")
    print(f"  {'─'*40}")
    print(f"\n  What these mean:")
    print(f"  Accuracy  = Overall correct decisions")
    print(f"  Precision = When AI says attack, how often right?")
    print(f"  Recall    = Of all real attacks, how many caught?")
    print(f"  F1-Score  = Balance between precision and recall")
    print("=" * 60)

    return {
        'accuracy': accuracy, 'precision': precision,
        'recall': recall,     'f1': f1,
        'TP': TP, 'FP': FP,  'TN': TN, 'FN': FN
    }


# ═══════════════════════════════════════════════════════
# PART 3: DRAW TRAINING PROGRESS CHARTS
# ═══════════════════════════════════════════════════════

def plot_training(history):
    """
    Draw 4 charts showing how the AI improved over time.
    Saved as training_results.png
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Progress — Adaptive Cyber Threat Detection',
                 fontsize=14, fontweight='bold')

    episodes = history['episode']
    colors   = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    titles   = ['Detection Rate (%)', 'Accuracy (%)',
                'Total Reward',       'Loss']
    keys     = ['detection_rate', 'accuracy', 'total_reward', 'avg_loss']

    for ax, key, title, color in zip(axes.flat, keys, titles, colors):
        ax.plot(episodes, history[key],
                color=color, linewidth=2, marker='o', markersize=5)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(episodes)

        # Add value labels on each point
        for x, y in zip(episodes, history[key]):
            ax.annotate(f'{y:.1f}',
                        (x, y), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✅ Chart saved as training_results.png")