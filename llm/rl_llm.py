from transformers import pipeline
import numpy as np

"""
LLM Prompt Optimization via Reward Feedback (RL-Inspired)

Built a lightweight decision-making system using a language model to generate candidate actions from prompts
Designed reward functions to evaluate outputs and select optimal responses
Simulated reinforcement learning concepts such as policy improvement through reward-based selection
"""



# Load small model (very fast)
# This creates a text-generation pipeline using the Hugging Face Transformers library
# pipeline is a high-level Hugging Face API for running pretrained models without manually implementing tokenization and inference.
# distilgpt2 is a smaller, faster version of GPT-2.It is an autoregressive transformer language model, meaning it generates text one token at a time.
generator = pipeline("text-generation", model="distilgpt2")
# DistilGPT2 is primarily a text-completion model, not an instruction-tuned model. Therefore, it may generate unrelated, incomplete, 
# or unsafe responses rather than clearly answering the question.
# -----------------------------
# Task: decision making
# -----------------------------
state = "The road is icy. What should you do?"
actions = ["slow down", "speed up", "turn sharply"]

# -----------------------------
# Reward function
# -----------------------------
def reward(text):
    # This function assigns a reward to each generated response
    if "slow" in text.lower()  or "reduce speed" in text.lower() :
        return 1.0
    return 0.0

# -----------------------------
# Generate candidates
# -----------------------------
# This function generates multiple responses to the same prompt.
def generate_candidates(prompt, n=5):
    #outputs = generator(prompt, max_length=40, num_return_sequences=n)
    outputs = generator(
        prompt,
        max_new_tokens=30, #max_new_tokens=30: generate at most 30 additional tokens
        do_sample=True, #sample tokens probabilistically
        temperature=0.8, #control randomness; lower temperatures generally produce more concentrated output distributions.
        num_return_sequences=n, #produce multiple candidates
        return_full_text=False, # the generated text also contains the original prompt. If the prompt itself contains the word "slow", every candidate will receive a reward of 1.0 regardless of the model's continuation.
                                #void this by using return_full_text=False when calling the generator.
        pad_token_id=generator.tokenizer.eos_token_id #avoid common GPT-2 padding warnings
    )
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
print(f"Text: {best[0]}")
print(f"Reward: {best[1]}")