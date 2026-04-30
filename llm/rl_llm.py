from transformers import pipeline
import numpy as np

"""
LLM Prompt Optimization via Reward Feedback (RL-Inspired)

Built a lightweight decision-making system using a language model to generate candidate actions from prompts
Designed reward functions to evaluate outputs and select optimal responses
Simulated reinforcement learning concepts such as policy improvement through reward-based selection
"""



# Load small model (very fast)
generator = pipeline("text-generation", model="distilgpt2")

# -----------------------------
# Task: decision making
# -----------------------------
state = "The road is icy. What should you do?"
actions = ["slow down", "speed up", "turn sharply"]

# -----------------------------
# Reward function
# -----------------------------
def reward(text):
    if "slow" in text.lower():
        return 1.0
    return 0.0

# -----------------------------
# Generate candidates
# -----------------------------
def generate_candidates(prompt, n=5):
    outputs = generator(prompt, max_length=40, num_return_sequences=n)
    return [o["generated_text"] for o in outputs]

# -----------------------------
# RL-style selection
# -----------------------------
def select_best(prompt):
    candidates = generate_candidates(prompt)

    scored = []
    for c in candidates:
        r = reward(c)
        scored.append((c, r))

    best = max(scored, key=lambda x: x[1])
    return best, scored

# -----------------------------
# Run
# -----------------------------
best, all_outputs = select_best(state)

print("All candidates:")
for text, r in all_outputs:
    print(f"Reward={r:.1f} | {text}\n")

print("Best choice:")
print(best)