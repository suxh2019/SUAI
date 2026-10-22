"""Educational CPU-only demonstration: language-model pretraining then Supervised Fine-Tuning(SFT).
Install: pip install torch
Run: python llm_post_training_sft_cpu.py
This trains a tiny CHARACTER-level transformer; it is not a production LLM.
"""
import math
import random
import torch
from torch import nn
import torch.nn.functional as F

# Make the experiment reproducible on the same hardware/software setup.
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# Force CPU execution, including on laptops without a GPU.
DEVICE = torch.device('cpu')
torch.set_num_threads(min(4, torch.get_num_threads()))

# Small model/training settings for a laptop.
CONTEXT = 96            # Maximum input sequence length in characters.
DIM = 48                # Number of embedding features.
HEADS = 4               # Number of attention heads.
LAYERS = 2              # Number of transformer blocks.
BATCH_SIZE = 16         # Examples in each optimizer update.
PRETRAIN_STEPS = 160    # Reduce for a faster smoke test.
SFT_STEPS = 250         # Increase to learn the tiny instruction dataset better.
LEARNING_RATE = 0.001   # AdamW learning rate.

# General text for the initial next-character prediction stage.
pretrain_lines = [
    'The sun rises in the east. The sky is blue.',
    'A cat is an animal. A dog is an animal.',
    'Two plus two equals four. Three plus three equals six.',
    'Red, green, and blue are colors.',
    'A bicycle has two wheels. A car has four wheels.',
    'A language model predicts the next token in a sequence.',
    'Machine learning systems learn patterns from data.',
]

# SFT examples: instruction/answer pairs; examples are intentionally repetitive.
sft_pairs = [
    ('What is 2 + 2?', '4'),
    ('What is 3 + 3?', '6'),
    ('What color is the sky?', 'blue'),
    ('What animal says meow?', 'cat'),
    ('What animal barks?', 'dog'),
    ('How many wheels does a bicycle have?', '2'),
    ('Say hello.', 'Hello!'),
]

# A consistent prompt template makes instruction tuning more learnable.
def make_prompt(question):
    return 'Question: ' + question + '\nAnswer:'

# Build a character vocabulary using all text needed in both phases.
all_text = '\n'.join(pretrain_lines)
for question, answer in sft_pairs:
    all_text += '\n' + make_prompt(question) + ' ' + answer + '\n'
chars = sorted(set(all_text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
VOCAB = len(chars)

# Map text to token IDs (one token per character).
def encode(text):
    return [stoi[ch] for ch in text]

# Convert model IDs back into human-readable characters.
def decode(ids):
    return ''.join(itos[int(i)] for i in ids)

# Decoder-only GPT-style network with causal attention.
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB, DIM)             # Character embeddings.
        self.pos_emb = nn.Embedding(CONTEXT, DIM)            # Position embeddings.
        layer = nn.TransformerEncoderLayer(                 # One transformer block.
            d_model=DIM, nhead=HEADS, dim_feedforward=4*DIM,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, LAYERS)  # Stack of blocks.
        self.norm = nn.LayerNorm(DIM)                       # Normalize hidden states.
        self.head = nn.Linear(DIM, VOCAB)                   # Predict next character.

    def forward(self, ids):
        B, T = ids.shape                                    # Batch and sequence length.
        positions = torch.arange(T, device=ids.device)     # Indices 0, 1, ..., T-1.
        x = self.token_emb(ids) + self.pos_emb(positions)   # Add token + position data.
        causal_mask = torch.triu(                           # Prevent future-token access.
            torch.ones(T, T, device=ids.device, dtype=torch.bool), diagonal=1
        )
        x = self.blocks(x, mask=causal_mask)                # Causal self-attention.
        return self.head(self.norm(x))                      # Logits [B, T, VOCAB].

# Create the model on the CPU.
model = TinyGPT().to(DEVICE)
print('Trainable parameters:', sum(p.numel() for p in model.parameters()))

# PRETRAINING: sample sequences from repeated general-purpose text.
stream = encode(('\n'.join(pretrain_lines) + '\n') * 200)

def pretrain_batch():
    xs, ys = [], []
    for _ in range(BATCH_SIZE):
        start = random.randrange(len(stream) - CONTEXT - 1)
        chunk = stream[start:start + CONTEXT + 1]
        xs.append(chunk[:-1])                             # Input characters.
        ys.append(chunk[1:])                              # Next-character targets.
    return torch.tensor(xs, device=DEVICE), torch.tensor(ys, device=DEVICE)

# Create a new optimizer for pretraining.
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
print('\nPhase 1: pretraining on general text')
model.train()
for step in range(1, PRETRAIN_STEPS + 1):
    x, y = pretrain_batch()                                # Sample training examples.
    logits = model(x)                                      # Forward pass.
    loss = F.cross_entropy(                                # Next-token prediction loss.
        logits.reshape(-1, VOCAB), y.reshape(-1)
    )
    optimizer.zero_grad()                                  # Erase old gradients.
    loss.backward()                                        # Backpropagation.
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)       # Limit gradient magnitude.
    optimizer.step()                                       # Update model weights.
    if step % 40 == 0:
        print(f'pretrain step {step:3d} | loss {loss.item():.3f}')

# Generate text one character at a time.
@torch.no_grad()
def generate(prompt, new_chars=24):
    model.eval()                                           # Use inference mode.
    ids = encode(prompt)                                   # Encode instruction.
    for _ in range(new_chars):
        current = torch.tensor([ids[-CONTEXT:]], device=DEVICE)
        logits = model(current)[0, -1]                     # Last-position logits.
        next_id = int(torch.argmax(logits).item())         # Greedy decoding.
        ids.append(next_id)                                # Append prediction.
        if itos[next_id] == '\n':                          # Stop when answer ends.
            break
    return decode(ids[len(encode(prompt)):]).strip()        # Return generated suffix.

question = 'What is 2 + 2?'
print('\nBefore SFT:', repr(generate(make_prompt(question) + ' ')))

# SUPERVISED FINE-TUNING: only ANSWER tokens contribute to the loss.
def make_sft_example(question, answer):
    prefix = make_prompt(question) + ' '                    # Instruction and answer marker.
    complete = prefix + answer + '\n'                     # Full training string.
    ids = encode(complete)                                 # Character-token IDs.
    if len(ids) > CONTEXT + 1:
        raise ValueError('Increase CONTEXT for longer examples.')
    inputs = ids[:-1]                                      # Every token except the last.
    labels = ids[1:]                                       # Predict the following token.
    # Labels at positions before the first answer character are ignored.
    # For shifted targets, the answer starts at index len(prefix)-1.
    labels[:len(prefix)-1] = [-100] * (len(prefix)-1)
    return inputs, labels

examples = [make_sft_example(q, a) for q, a in sft_pairs]

# Build minibatches padded to their longest sequence; ignore padding in loss.
def sft_batch():
    chosen = random.choices(examples, k=BATCH_SIZE)        # Sample examples with replacement.
    length = max(len(x) for x, _ in chosen)                 # Longest example this batch.
    x_batch = torch.zeros((BATCH_SIZE, length), dtype=torch.long)
    y_batch = torch.full((BATCH_SIZE, length), -100, dtype=torch.long)
    for i, (x, y) in enumerate(chosen):
        x_batch[i, :len(x)] = torch.tensor(x)               # Place real tokens.
        y_batch[i, :len(y)] = torch.tensor(y)               # Ignore padding via -100.
    return x_batch.to(DEVICE), y_batch.to(DEVICE)

# Continue training the SAME model with a fresh optimizer for the new phase.
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE / 2)
print('\nPhase 2: supervised instruction fine-tuning')
model.train()
for step in range(1, SFT_STEPS + 1):
    x, y = sft_batch()                                     # Instructions and answer labels.
    logits = model(x)                                      # Forward pass.
    loss = F.cross_entropy(                                # Loss only on answer characters.
        logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100
    )
    optimizer.zero_grad()                                  # Reset accumulated gradients.
    loss.backward()                                        # Compute new gradients.
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)       # Stabilize optimization.
    optimizer.step()                                       # Fine-tune model parameters.
    if step % 50 == 0:
        print(f'SFT step {step:3d} | answer loss {loss.item():.3f}')

# Compare answers on the small set of training instructions.
print('\nAfter SFT (answers to training instructions):')
for q, target in sft_pairs:
    print(f'Q: {q}\nExpected: {target} | Generated: {generate(make_prompt(q) + ' ')!r}\n')

# Save learned weights and the character vocabulary for future loading.
torch.save({
    'model_state': model.state_dict(),
    'vocab': chars,
    'settings': {'context': CONTEXT, 'dim': DIM, 'heads': HEADS, 'layers': LAYERS},
}, 'tiny_gpt_after_sft.pt')
print('Saved model to tiny_gpt_after_sft.pt')
