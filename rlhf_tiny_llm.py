"""Laptop-sized RLHF （Reinforcement Learning from Human Feedback (RLHF)） 
demonstration: SFT -> preference reward model -> PPO-style RL.

Install: python -m pip install torch
Run:     python rlhf_tiny_llm_cpu.py

This is an educational *single-token* response model, not production multi-token RLHF.
The preferences are generated synthetically from known arithmetic answers instead of humans.
"""

import copy  # Make a frozen reference copy of the supervised policy.
import random  # Make reproducible synthetic preference comparisons.
import torch  # Implement tensor computations and automatic differentiation.
from torch import nn  # Build the neural networks.
from torch.nn import functional as F  # Compute cross-entropy and other losses.

# Fix the random seed so runs are easier to compare.
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Limit CPU threads so a very small model avoids excessive thread overhead.
torch.set_num_threads(1)

# The script intentionally runs on a CPU-only laptop.
DEVICE = torch.device("cpu")

# Each prompt is an addition question with a one-digit answer.
QUESTIONS = [(a, b) for a in range(5) for b in range(5)]

# Use character tokenization for a minimal, self-contained Transformer.
CHARS = list("0123456789+= ")
STOI = {character: index for index, character in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)

# Candidate answers are exactly the digits 0 through 9.

def encode_prompt(a, b):  # Convert a math question into character IDs.
    text = f"{a}+{b}="  # Format a question, for example, '2+3='.
    return torch.tensor([STOI[c] for c in text], dtype=torch.long, device=DEVICE)


# Stack every question into a single fixed-size tensor of shape [25, 4].
PROMPTS = torch.stack([encode_prompt(a, b) for a, b in QUESTIONS])

# Correct labels are the sums, each an integer in 0,...,8.
LABELS = torch.tensor([a + b for a, b in QUESTIONS], dtype=torch.long, device=DEVICE)


class TinyPolicy(nn.Module):  # Causal Transformer predicting a one-token answer.
    def __init__(self, width=32, heads=4):  # Keep the model small for the CPU.
        super().__init__()  # Initialize nn.Module.
        self.token_emb = nn.Embedding(VOCAB_SIZE, width)  # Learn character vectors.
        self.pos_emb = nn.Embedding(4, width)  # Learn the four input positions.
        layer = nn.TransformerEncoderLayer(  # Define one self-attention block.
            d_model=width, nhead=heads, dim_feedforward=64,
            dropout=0.0, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)  # One block.
        self.out = nn.Linear(width, 10)  # One logit for each possible answer digit.

    def forward(self, prompts):  # Return action logits for a batch of questions.
        batch, length = prompts.shape  # Read batch size and sequence length.
        positions = torch.arange(length, device=prompts.device)  # 0,1,2,3.
        hidden = self.token_emb(prompts) + self.pos_emb(positions)  # Embed inputs.
        causal_mask = torch.triu(  # Block attention to future characters.
            torch.ones(length, length, device=prompts.device, dtype=torch.bool),
            diagonal=1
        )
        hidden = self.transformer(hidden, mask=causal_mask)  # Run attention.
        return self.out(hidden[:, -1])  # Predict the answer after the '=' token.


class RewardModel(nn.Module):  # Model a learned scalar preference score.
    def __init__(self, width=32):  # Use a small embedding-based reward model.
        super().__init__()  # Initialize PyTorch module tracking.
        self.question_emb = nn.Embedding(VOCAB_SIZE, width)  # Embed prompt tokens.
        self.answer_emb = nn.Embedding(10, width)  # Embed candidate answer digit.
        self.net = nn.Sequential(  # Score the concatenated representations.
            nn.Linear(width * 2, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, prompts, answers):  # Output one score per prompt/answer.
        # Average over the four tokens to obtain a fixed width representation.
        representation = self.question_emb(prompts).mean(dim=1)  # [B, width].
        response = self.answer_emb(answers)  # [B, width].
        features = torch.cat([representation, response], dim=-1)  # [B, 2*width].
        return self.net(features).squeeze(-1)  # [B] learned preference score.


def evaluate(policy):  # Measure greedy accuracy on the 25 known questions.
    policy.eval()  # Disable training behavior (such as dropout).
    with torch.no_grad():  # Inference does not need gradients.
        predictions = policy(PROMPTS).argmax(dim=-1)  # Pick the largest logit.
        accuracy = (predictions == LABELS).float().mean().item()  # Fraction right.
    return accuracy  # Return a Python float between zero and one.


def train_sft(policy, steps=25):  # Warm-start with supervised fine-tuning.
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)  # Adam optimizer.
    policy.train()  # Enable training mode.
    for step in range(steps):  # Repeat gradient updates.
        logits = policy(PROMPTS)  # Predict answer probabilities from prompts.
        loss = F.cross_entropy(logits, LABELS)  # Imitate known correct answers.
        optimizer.zero_grad()  # Discard the last iteration's gradients.
        loss.backward()  # Differentiate SFT cross-entropy through the model.
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # Limit large gradients.
        optimizer.step()  # Apply the parameter update.
        if (step + 1) % 25 == 0:  # Print occasional progress information.
            print(f"SFT step {step+1:3d} | loss={loss.item():.3f} | accuracy={evaluate(policy):.1%}")
            policy.train()  # Return to train mode after the evaluation.


def train_reward_model(reward_model, steps=250):  # Learn synthetic preferences.
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.003)  # Optimizer.
    for step in range(steps):  # Perform preference-learning updates.
        indices = torch.randint(0, len(QUESTIONS), (64,))  # Sample questions.
        prompts = PROMPTS[indices]  # Get the corresponding encoded questions.
        chosen = LABELS[indices]  # Synthetic 'human preferred' correct answers.
        offsets = torch.randint(1, 10, chosen.shape)  # Sample nonzero offsets.
        rejected = (chosen + offsets) % 10  # Always select an incorrect answer.
        good_scores = reward_model(prompts, chosen)  # Score preferred answers.
        bad_scores = reward_model(prompts, rejected)  # Score rejected answers.
        # Bradley-Terry preference loss: prefer chosen scores over rejected.
        loss = -F.logsigmoid(good_scores - bad_scores).mean()
        optimizer.zero_grad()  # Reset reward-model gradients.
        loss.backward()  # Differentiate the pairwise ranking objective.
        optimizer.step()  # Update reward-model parameters.
        if (step + 1) % 50 == 0:  # Show reward-model learning progress.
            print(f"Reward step {step+1:3d} | preference loss={loss.item():.3f}")
    reward_model.eval()  # Fix reward model during RL policy optimization.
    for parameter in reward_model.parameters():  # Stop reward-model training.
        parameter.requires_grad_(False)  # Policy training must not change rewards.


def train_rlhf(policy, reward_model, steps=150, group=8):  # PPO-style RLHF.
    reference = copy.deepcopy(policy).eval()  # Freeze the SFT reference policy.
    for parameter in reference.parameters():  # Freeze its trainable weights.
        parameter.requires_grad_(False)  # No reference optimizer updates.
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0005)  # Smaller RL LR.
    clip_epsilon = 0.2  # PPO probability-ratio clipping range.
    kl_beta = 0.02  # Penalize divergence from the SFT reference policy.
    for step in range(steps):  # Repeat rollout-and-update cycles.
        indices = torch.randint(0, len(QUESTIONS), (16,))  # Sample prompt batch.
        prompts = PROMPTS[indices]  # Gather sampled prompts.
        with torch.no_grad():  # Rollouts are generated by the old policy.
            old_logits = policy(prompts)  # Snapshot sampling logits.
            old_dist = torch.distributions.Categorical(logits=old_logits)  # Policy.
            samples = old_dist.sample((group,))  # [group, B] sampled outputs.
            old_logp = old_dist.log_prob(samples).transpose(0, 1).detach()  # [B, group].
            actions = samples.transpose(0, 1).contiguous()  # [B, group].
            repeated = prompts.repeat_interleave(group, dim=0)  # Repeat prompts.
            scores = reward_model(repeated, actions.reshape(-1))  # RM reward.
            scores = scores.reshape(-1, group)  # [B, group] reward groups.
            # A within-prompt baseline compares responses for the same question.
            advantages = scores - scores.mean(dim=1, keepdim=True)
            # Standardize advantages to prevent wildly varying gradient scales.
            advantages = advantages / (scores.std(unbiased=False) + 1e-6)
        # Perform two PPO epochs over these fixed sampled actions.
        for _ in range(2):  # Reuse samples with a clipped importance ratio.
            new_logits = policy(prompts)  # Compute updated logits.
            new_dist = torch.distributions.Categorical(logits=new_logits)  # π_new.
            new_logp = new_dist.log_prob(actions.transpose(0, 1)).transpose(0, 1)  # [B, group].
            ratio = torch.exp(new_logp - old_logp)  # π_new(a)/π_old(a).
            surrogate = ratio * advantages  # Unclipped policy objective.
            clipped = ratio.clamp(1-clip_epsilon, 1+clip_epsilon) * advantages
            policy_loss = -torch.minimum(surrogate, clipped).mean()  # PPO loss.
            with torch.no_grad():  # Reference distribution stays frozen.
                reference_logits = reference(prompts)  # SFT reference logits.
            log_new = F.log_softmax(new_logits, dim=-1)  # Log π_new for all digits.
            log_ref = F.log_softmax(reference_logits, dim=-1)  # Log π_ref.
            probs = log_new.exp()  # Convert log probabilities to probabilities.
            kl = (probs * (log_new - log_ref)).sum(dim=-1).mean()  # Exact KL.
            loss = policy_loss + kl_beta * kl  # Clipped PPO + KL penalty.
            optimizer.zero_grad()  # Reset current policy gradients.
            loss.backward()  # Differentiate the surrogate objective.
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)  # Clip gradients.
            optimizer.step()  # Improve the policy using reward feedback.
        if (step + 1) % 25 == 0:  # Report training behavior.
            print(f"RLHF step {step+1:3d} | loss={loss.item():+.3f} "
                  f"| KL={kl.item():.3f} | accuracy={evaluate(policy):.1%}")


def main():  # Execute all three phases in order.
    print("Educational tiny Transformer RLHF on CPU")  # Identify the demo.
    policy = TinyPolicy().to(DEVICE)  # Construct the language policy.
    reward_model = RewardModel().to(DEVICE)  # Construct a learned reward model.
    print(f"Initial accuracy: {evaluate(policy):.1%}")  # Before training.
    train_sft(policy)  # Phase 1: supervised training.
    print(f"After SFT: {evaluate(policy):.1%}")  # Measure SFT performance.
    train_reward_model(reward_model)  # Phase 2: preference reward modeling.
    train_rlhf(policy, reward_model)  # Phase 3: PPO-style RLHF training.
    print(f"Final accuracy: {evaluate(policy):.1%}")  # Evaluate final policy.
    with torch.no_grad():  # Print a few concrete model answers.
        for a, b in [(1, 2), (2, 3), (4, 4)]:  # Example questions.
            answer = policy(encode_prompt(a, b).unsqueeze(0)).argmax().item()
            print(f"{a}+{b} = {answer} (correct: {a+b})")  # Show prediction.
    torch.save(policy.state_dict(), "rlhf_tiny_policy.pt")  # Save parameters.
    print("Saved model to rlhf_tiny_policy.pt")  # Confirm file destination.


if __name__ == "__main__":  # Run only when invoked as a program.
    main()  # Start the full RLHF pipeline.
