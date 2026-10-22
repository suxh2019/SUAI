"""Train a tiny GPT-style character language model on a laptop CPU.

Install: pip install torch
Run:     python train_tiny_llm_cpu.py

This teaches next-character prediction from scratch. It is an educational
transformer, not a pretrained assistant or an instruction-following LLM.
"""

'''
Key takeaway: This example teaches how GPT-style language model pretraining works: 
convert text into tokens, compute causal attention, predict the next token, minimize 
cross-entropy, and update neural network parameters through backpropagation.
'''

import math  # Provides the square-root scaling used by attention.
import torch  # Provides tensors, neural network operations, and training.
import torch.nn as nn  # Provides neural network layers.
import torch.nn.functional as F  # Provides loss and activation functions.

# Fix the random seed for reproducible initial model weights and sampling.
torch.manual_seed(42)
# Force computations to run on the CPU, so no GPU is necessary.
device = torch.device("cpu")
# Limit CPU threads to keep this tiny example friendly to ordinary laptops.
torch.set_num_threads(min(4, torch.get_num_threads()))

# Set a small batch size to limit memory usage.
BATCH_SIZE = 16
# Each training sample contains this many input characters.
BLOCK_SIZE = 64
# Number of training gradient updates; increase to improve learning.
TRAIN_STEPS = 400
# Print training loss at this interval.
PRINT_EVERY = 50
# Size of each token embedding vector.
EMBED_DIM = 64
# Number of attention heads in each transformer block.
NUM_HEADS = 4
# Number of transformer blocks.
NUM_LAYERS = 2
# Optimizer learning rate.
LEARNING_RATE = 3e-4

# A small built-in training corpus keeps the example independent of downloads.
# Replace this with text from a UTF-8 file for more interesting experiments.
base_text = (
    "A tiny language model learns to predict the next character.\n"
    "Machine learning uses examples to update model parameters.\n"
    "A neural network makes predictions from its input context.\n"
    "Transformers use attention to relate earlier tokens to later tokens.\n"
    "A model trained on a small corpus will mostly imitate that corpus.\n"
    "The quick brown fox jumps over the lazy dog.\n"
    "The road is icy, so drive slowly and avoid sharp turns.\n"
    "Reinforcement learning learns policies from rewards.\n"
)
# Repeat the small collection of sentences to form enough training examples.
text = base_text * 100
# Build a sorted list of unique characters occurring in the corpus.
chars = sorted(set(text))
# Count the number of distinct characters the model can predict.
vocab_size = len(chars)
# Map each character to an integer token ID.
stoi = {char: idx for idx, char in enumerate(chars)}
# Map integer token IDs back to characters for text generation.
itos = {idx: char for char, idx in stoi.items()}
# Convert all training text into integer character IDs.
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

# Define a function that samples a random batch of sequences.
def get_batch():
    # Choose random starting positions that leave room for the next-character labels.
    starts = torch.randint(0, len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    # Build a batch of input token sequences.
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in starts.tolist()])
    # Build targets shifted one character to the right.
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in starts.tolist()])
    # Return inputs and next-character targets on the chosen device.
    return x.to(device), y.to(device)

# Define a transformer block with causal self-attention and an MLP.
class TransformerBlock(nn.Module):
    # Configure its components when it is created.
    def __init__(self):
        # Initialize PyTorch's parent module.
        super().__init__()
        # Normalize token representations before attention.
        self.norm1 = nn.LayerNorm(EMBED_DIM)
        # Use PyTorch multi-head attention, operating on batch-first tensors.
        self.attn = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=True)
        # Normalize token representations before the feed-forward network.
        self.norm2 = nn.LayerNorm(EMBED_DIM)
        # Mix features independently at each token position.
        self.mlp = nn.Sequential(
            nn.Linear(EMBED_DIM, 4 * EMBED_DIM),  # Expand feature dimension.
            nn.GELU(),  # Apply a smooth nonlinear activation.
            nn.Linear(4 * EMBED_DIM, EMBED_DIM),  # Restore the embedding dimension.
        )

    # Define one forward pass through the transformer block.
    def forward(self, x):
        # Read the number of tokens in the current context.
        T = x.size(1)
        # Block future tokens so predictions can only use earlier tokens.
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        # Normalize the inputs to the attention layer.
        h = self.norm1(x)
        # Compute causal self-attention; attention weights need not be returned.
        attended, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        # Add a residual connection to preserve the previous features.
        x = x + attended
        # Apply the feed-forward network with another residual connection.
        x = x + self.mlp(self.norm2(x))
        # Return the updated representations.
        return x

# Define the entire autoregressive GPT-style language model.
class TinyGPT(nn.Module):
    # Create embeddings, transformer blocks, and the output layer.
    def __init__(self):
        # Initialize PyTorch's parent module.
        super().__init__()
        # Convert token IDs into learnable vectors.
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)
        # Give every context position its own learned position vector.
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM)
        # Stack the transformer blocks to process token context.
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_LAYERS)])
        # Normalize the final representations.
        self.final_norm = nn.LayerNorm(EMBED_DIM)
        # Convert representations into a logit for each character.
        self.output = nn.Linear(EMBED_DIM, vocab_size)

    # Compute unnormalized next-character scores for every input position.
    def forward(self, tokens):
        # Extract batch size and sequence length.
        B, T = tokens.shape
        # Reject inputs longer than the model's supported context window.
        assert T <= BLOCK_SIZE, "Input exceeds the model context length"
        # Create integer positions from 0 through T-1.
        positions = torch.arange(T, device=tokens.device)
        # Add token embeddings and positional embeddings together.
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        # Process the sequence through the attention/MLP blocks.
        x = self.blocks(x)
        # Normalize and project to vocabulary logits.
        return self.output(self.final_norm(x))

    # Generate new characters after a starting context.
    @torch.no_grad()  # Disable gradients, because generation is inference only.
    def generate(self, context, max_new_tokens=120):
        # Switch dropout and other relevant modules into evaluation mode.
        self.eval()
        # Generate one new character per iteration.
        for _ in range(max_new_tokens):
            # Keep only the most recent positions supported by the context window.
            context_window = context[:, -BLOCK_SIZE:]
            # Predict character logits for the current context.
            logits = self(context_window)
            # Take logits from the final position only.
            next_logits = logits[:, -1, :]
            # Convert logits to a probability distribution.
            probabilities = F.softmax(next_logits, dim=-1)
            # Sample the next character ID from that distribution.
            next_token = torch.multinomial(probabilities, num_samples=1)
            # Append the sampled character to the growing text.
            context = torch.cat([context, next_token], dim=1)
        # Return all original and generated tokens.
        return context

# Construct the model and move it to the CPU.
model = TinyGPT().to(device)
# Create AdamW to update the model's trainable weights.
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# Display the number of learned parameters.
print(f"Vocabulary size: {vocab_size}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

# Put the model in training mode.
model.train()
# Repeat gradient descent for the requested number of steps.
for step in range(1, TRAIN_STEPS + 1):
    # Sample random input sequences and next-character labels.
    inputs, labels = get_batch()
    # Compute the model's logits, shaped [batch, time, vocabulary].
    logits = model(inputs)
    # Flatten the predictions and labels for cross-entropy.
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1))
    # Clear gradients accumulated from the previous optimization step.
    optimizer.zero_grad()
    # Backpropagate the loss to calculate gradients.
    loss.backward()
    # Clip gradients to help avoid unstable updates.
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # Update model parameters with AdamW.
    optimizer.step()
    # Print progress periodically and at the first step.
    if step == 1 or step % PRINT_EVERY == 0:
        print(f"Step {step:4d}/{TRAIN_STEPS} | Training loss: {loss.item():.4f}", flush=True)

# Save model weights and the character vocabulary for later use.
torch.save({"model_state": model.state_dict(), "chars": chars}, "tiny_gpt_cpu.pt")
# Choose a starting prompt that appears in the training vocabulary.
prompt = "A neural network"
# Convert the prompt into a 1-row tensor of token IDs.
context = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
# Generate 150 additional characters after the prompt.
generated_ids = model.generate(context, max_new_tokens=150)[0].tolist()
# Decode token IDs back into a printable string.
generated_text = "".join(itos[token_id] for token_id in generated_ids)
# Display the final generated text.
print("\nGenerated text:\n" + generated_text)
# Confirm where the model checkpoint was saved.
print("\nSaved checkpoint: tiny_gpt_cpu.pt")
