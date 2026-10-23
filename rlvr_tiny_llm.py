"""Laptop-friendly reinforcement learning with verifiable rewards (RLVR).

A tiny causal Transformer produces ONE answer token (0..9) to an addition prompt.
The exact arithmetic answer supplies an automatically verifiable 0/1 reward.
After supervised warm-up, a GRPO-style grouped policy-gradient update improves
its policy without training a separate value network. CPU-only, PyTorch only.
"""

import copy  # Make a frozen reference policy for KL regularization.
import random  # Seed the built-in Python random generator.
import torch  # Tensor operations and automatic differentiation.
import torch.nn as nn  # Neural network layers.
import torch.nn.functional as F  # Cross entropy and other operations.

# Make repeated runs more reproducible on the same machine.
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)  # Avoid CPU oversubscription for this tiny model.
DEVICE = torch.device('cpu')  # Require no GPU or CUDA.

# Use a restricted arithmetic task so answers are single digit tokens.
PAIRS = [(a, b) for a in range(5) for b in range(5)]  # 25 questions, sums 0..8.
CHARS = list('0123456789+= ')  # Fixed tiny character vocabulary.
STOI = {ch: i for i, ch in enumerate(CHARS)}  # Character -> token ID.
VOCAB = len(CHARS)  # Number of possible input tokens.
ANSWER_TOKENS = torch.tensor([STOI[str(i)] for i in range(9)], device=DEVICE)  # Allowed answer tokens.

# Convert each question, such as '2+3=', into token IDs.
def encode_prompt(a, b):
    return [STOI[ch] for ch in f'{a}+{b}=']

PROMPTS = torch.tensor([encode_prompt(a, b) for a, b in PAIRS], dtype=torch.long, device=DEVICE)
LABELS = torch.tensor([a + b for a, b in PAIRS], dtype=torch.long, device=DEVICE)

# Build a small causal Transformer policy (not a pretrained commercial LLM).
class TinyPolicy(nn.Module):
    def __init__(self, width=32, heads=4):
        super().__init__()  # Initialize base PyTorch module.
        self.token_embedding = nn.Embedding(VOCAB, width)  # Embed input tokens.
        self.position_embedding = nn.Embedding(4, width)  # Four prompt positions.
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)  # Self-attention.
        self.norm1 = nn.LayerNorm(width)  # Stabilize attention residual.
        self.ffn = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, width))  # MLP block.
        self.norm2 = nn.LayerNorm(width)  # Stabilize MLP residual.
        self.head = nn.Linear(width, 9)  # Predict one answer among digits 0..8.

    def forward(self, prompts):
        batch, length = prompts.shape  # Batch size and token sequence length.
        positions = torch.arange(length, device=prompts.device)  # Position indices.
        x = self.token_embedding(prompts) + self.position_embedding(positions)  # Embed text and position.
        future_mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=prompts.device), diagonal=1)  # Causal mask.
        attended, _ = self.attention(x, x, x, attn_mask=future_mask, need_weights=False)  # Self-attention.
        x = self.norm1(x + attended)  # Residual connection.
        x = self.norm2(x + self.ffn(x))  # Feed-forward residual connection.
        return self.head(x[:, -1, :])  # Last position predicts next answer token.

# Evaluate exact-answer accuracy on the 25 available questions.
@torch.no_grad()
def accuracy(policy):
    policy.eval()  # Disable training-only behaviors.
    answers = policy(PROMPTS).argmax(dim=-1)  # Greedy answer 0..8.
    return (answers == LABELS).float().mean().item()  # Mean exact-match accuracy.

# Initialize the model and supervised optimizer.
policy = TinyPolicy().to(DEVICE)
optimizer = torch.optim.AdamW(policy.parameters(), lr=0.003)
print(f'Initial accuracy: {accuracy(policy):.1%}', flush=True)

# Stage 1: small supervised warm-up to create a reasonable initial policy.
SFT_STEPS = 40
for step in range(SFT_STEPS):
    policy.train()  # Enable parameter training.
    logits = policy(PROMPTS)  # Predict each correct next answer.
    loss = F.cross_entropy(logits, LABELS)  # Supervised next-token objective.
    optimizer.zero_grad()  # Clear old gradients.
    loss.backward()  # Compute derivatives.
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # Avoid gradient explosions.
    optimizer.step()  # Update model weights.
print(f'After supervised warm-up: {accuracy(policy):.1%}', flush=True)

# Keep the supervised initialization as the frozen KL reference.
reference = copy.deepcopy(policy).eval()  # Copy weights without sharing parameters.
for parameter in reference.parameters():
    parameter.requires_grad_(False)  # Never update the reference model.

# Stage 2: RLVR rewards are computed by an exact mathematical checker.
GROUP_SIZE = 8  # Responses sampled per question.
RL_STEPS = 180  # Number of offline question / policy optimization updates.
CLIP_EPS = 0.2  # Clipping radius around the old policy ratio.
KL_BETA = 0.02  # Penalty for moving far away from the reference policy.
rl_optimizer = torch.optim.AdamW(policy.parameters(), lr=0.0008)  # Smaller RL learning rate.

for step in range(1, RL_STEPS + 1):
    policy.train()  # Prepare the trainable actor.
    question_id = (step - 1) % len(PAIRS)  # Cycle through arithmetic questions.
    prompt = PROMPTS[question_id:question_id + 1]  # Shape (1, 4).
    correct = LABELS[question_id]  # Known ground-truth arithmetic result.

    with torch.no_grad():  # Sampling does not need a computation graph.
        old_logits = policy(prompt).squeeze(0)  # Old logits over answer digits.
        old_log_probs = F.log_softmax(old_logits, dim=-1)  # Old log probabilities.
        candidates = torch.distributions.Categorical(logits=old_logits).sample((GROUP_SIZE,))  # Group of sampled responses.
        sampled_old_log_probs = old_log_probs[candidates]  # Log probability for sampled responses.
        rewards = (candidates == correct).float()  # Verifiable reward: 1 correct, 0 incorrect.
        reward_std = rewards.std(unbiased=False)  # Within-group dispersion.
        advantages = (rewards - rewards.mean()) / (reward_std + 1e-5)  # Group-relative advantages.
        reference_log_probs = F.log_softmax(reference(prompt).squeeze(0), dim=-1)  # Frozen reference.

    # Recompute the new policy distribution with gradients enabled.
    new_logits = policy(prompt).squeeze(0)  # Differentiable policy logits.
    new_log_probs = F.log_softmax(new_logits, dim=-1)  # Differentiable log probabilities.
    sampled_new_log_probs = new_log_probs[candidates]  # Log probabilities of sampled answers.
    ratio = (sampled_new_log_probs - sampled_old_log_probs).exp()  # New / old probability ratio.
    unclipped = ratio * advantages  # Standard policy gradient surrogate.
    clipped = ratio.clamp(1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages  # PPO-style clipping.
    policy_loss = -torch.minimum(unclipped, clipped).mean()  # Maximize conservative surrogate.

    new_probs = new_log_probs.exp()  # Distribution over all nine answers.
    kl = (new_probs * (new_log_probs - reference_log_probs)).sum()  # Exact categorical KL(new || reference).
    total_loss = policy_loss + KL_BETA * kl  # Regularized RLVR objective.
    rl_optimizer.zero_grad()  # Reset gradients before updating.
    total_loss.backward()  # Differentiate RLVR objective through the Transformer.
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # Clip gradients.
    rl_optimizer.step()  # Improve response probabilities.

    if step % 30 == 0:  # Periodically show progress.
        print(f'RLVR step {step:3d}: accuracy={accuracy(policy):.1%}, '
              f'group_reward={rewards.mean().item():.2f}, KL={kl.item():.4f}', flush=True)

# Evaluate again; these are TRAINING questions, not held-out generalization.
print(f'Final training-question accuracy: {accuracy(policy):.1%}', flush=True)
policy.eval()  # Switch to evaluation mode.
with torch.no_grad():  # No gradients needed for predictions.
    guesses = policy(PROMPTS).argmax(-1).tolist()  # Predict all 25 questions.
for (a, b), guess in zip(PAIRS[:10], guesses[:10]):
    print(f'{a}+{b}= {guess}  (correct: {a+b})')  # Show example answers.

torch.save(policy.state_dict(), 'rlvr_tiny_policy.pt')  # Save learned weights.
print('Saved model: rlvr_tiny_policy.pt')  # Confirm output path.
