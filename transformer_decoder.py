"""Tiny decoder-only Transformer trained from scratch on a laptop CPU.

A decoder-only Transformer is the architecture used by many 
autoregressive language models, including the GPT family, Llama, and other modern LLMs.

A decoder-only Transformer predicts the next token using only the previous tokens.

Run: python transformer_laptop.py
Install: python -m pip install torch
"""

import torch  # Import tensor operations and automatic differentiation.
import torch.nn as nn  # Import neural-network building blocks.
import torch.nn.functional as F  # Import softmax and cross-entropy functions.

torch.manual_seed(42)  # Make weight initialization and sampling reproducible.
torch.set_num_threads(2)  # Avoid excessive CPU threading on small workloads.
device = torch.device("cpu")  # Run exclusively on a laptop CPU.

# Use a tiny repeating corpus so the complete exercise does not require downloads.
text = ("transformers use attention to learn patterns in sequences. "
        "attention lets each token use previous tokens for context. "
        "machine learning models learn by minimizing prediction loss. ") * 35
chars = sorted(set(text))  # Build a character-level vocabulary.
vocab_size = len(chars)  # Count possible output characters.
stoi = {ch: i for i, ch in enumerate(chars)}  # Map characters to integer IDs.
itos = {i: ch for ch, i in stoi.items()}  # Map IDs back to characters.
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)  # Encode text as token IDs.

block_size = 32  # Maximum number of characters in each training context.
batch_size = 16  # Number of independent sequences in one training batch.
embed_dim = 32  # Hidden size used to represent each token.
num_heads = 4  # Number of attention heads.
num_layers = 2  # Number of Transformer blocks.
train_steps = 120  # Keep the example fast enough for a laptop CPU.
learning_rate = 0.003  # Set the AdamW learning rate.


def get_batch():  # Produce randomly sampled next-character prediction examples.
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))  # Choose start positions.
    x = torch.stack([data[i:i + block_size] for i in starts])  # Gather input token sequences.
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])  # Shift targets one token forward.
    return x.to(device), y.to(device)  # Return the tensors on our CPU device.


class CausalSelfAttention(nn.Module):  # Implement multi-head masked self-attention.
    def __init__(self):  # Create attention projection layers.
        super().__init__()  # Initialize the parent PyTorch module.
        assert embed_dim % num_heads == 0  # Each head must have an integer feature size.
        self.head_dim = embed_dim // num_heads  # Calculate features per head.
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)  # Predict query, key, value in one layer.
        self.out = nn.Linear(embed_dim, embed_dim)  # Combine head outputs.
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))  # Keep past/current tokens.
        self.register_buffer("mask", mask)  # Store the causal mask with the module (not trainable).

    def forward(self, x):  # Process input embeddings of shape [batch, time, features].
        B, T, C = x.shape  # Read batch size, sequence length, embedding dimension.
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # Split projected features into Q, K, and V.
        q = q.view(B, T, num_heads, self.head_dim).transpose(1, 2)  # Shape Q as [B, heads, T, head_dim].
        k = k.view(B, T, num_heads, self.head_dim).transpose(1, 2)  # Shape K the same way.
        v = v.view(B, T, num_heads, self.head_dim).transpose(1, 2)  # Shape V the same way.
        scores = q @ k.transpose(-2, -1) / (self.head_dim ** 0.5)  # Compute scaled attention scores.
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))  # Block attention to future tokens.
        weights = F.softmax(scores, dim=-1)  # Normalize scores into attention probabilities.
        attended = weights @ v  # Combine values using the attention probabilities.
        attended = attended.transpose(1, 2).contiguous().view(B, T, C)  # Recombine all heads.
        return self.out(attended)  # Return attention output in the original feature dimension.


class TransformerBlock(nn.Module):  # Create one standard decoder Transformer block.
    def __init__(self):  # Build attention, normalization, and feed-forward modules.
        super().__init__()  # Register submodules correctly with PyTorch.
        self.ln1 = nn.LayerNorm(embed_dim)  # Normalize before self-attention.
        self.attn = CausalSelfAttention()  # Perform causal self-attention.
        self.ln2 = nn.LayerNorm(embed_dim)  # Normalize before the feed-forward network.
        self.ffn = nn.Sequential(  # Define a position-wise feed-forward neural network.
            nn.Linear(embed_dim, 4 * embed_dim),  # Expand the hidden representation.
            nn.GELU(),  # Apply a smooth nonlinear activation.
            nn.Linear(4 * embed_dim, embed_dim),  # Project back to the embedding dimension.
        )  # Finish the feed-forward network.

    def forward(self, x):  # Apply the Transformer block to a batch of sequences.
        x = x + self.attn(self.ln1(x))  # Add attention output through a residual connection.
        x = x + self.ffn(self.ln2(x))  # Add feed-forward output through another residual connection.
        return x  # Return processed token representations.


class TinyTransformer(nn.Module):  # Build the full decoder-only language model.
    def __init__(self):  # Create embeddings, Transformer blocks, and output head.
        super().__init__()  # Initialize the PyTorch module.
        self.token_emb = nn.Embedding(vocab_size, embed_dim)  # Convert token IDs into learned vectors.
        self.pos_emb = nn.Embedding(block_size, embed_dim)  # Encode positions within the context window.
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(num_layers)])  # Stack blocks.
        self.ln_final = nn.LayerNorm(embed_dim)  # Normalize final token representations.
        self.head = nn.Linear(embed_dim, vocab_size)  # Predict logits for every possible next token.

    def forward(self, idx):  # Return logits for an integer token batch [B, T].
        B, T = idx.shape  # Read input batch size and sequence length.
        positions = torch.arange(T, device=idx.device)  # Build positional indices [0, ..., T-1].
        x = self.token_emb(idx) + self.pos_emb(positions)  # Combine word and position information.
        x = self.blocks(x)  # Apply stacked Transformer blocks.
        x = self.ln_final(x)  # Normalize outputs before prediction.
        return self.head(x)  # Return logits of shape [B, T, vocab_size].

    @torch.no_grad()  # Disable gradients when generating text.
    def generate(self, idx, new_tokens=100):  # Autoregressively generate additional characters.
        self.eval()  # Switch the model to evaluation mode.
        for _ in range(new_tokens):  # Add one token at a time.
            recent = idx[:, -block_size:]  # Keep only the latest context window.
            logits = self(recent)  # Predict logits for all positions in that context.
            next_logits = logits[:, -1, :]  # Use only the prediction at the final position.
            probs = F.softmax(next_logits, dim=-1)  # Convert the logits into a probability distribution.
            next_id = torch.multinomial(probs, num_samples=1)  # Sample the next character ID.
            idx = torch.cat([idx, next_id], dim=1)  # Append the sampled token to the sequence.
        return idx  # Return the original prompt followed by generated token IDs.


def main():  # Train and evaluate the example model.
    model = TinyTransformer().to(device)  # Create the Transformer on the CPU.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # Construct the optimizer.
    parameters = sum(p.numel() for p in model.parameters())  # Count trainable parameters.
    print(f"Vocabulary size: {vocab_size}; parameters: {parameters:,}")  # Show model size.
    model.train()  # Enable training behavior.
    for step in range(1, train_steps + 1):  # Repeatedly improve model parameters.
        x, y = get_batch()  # Sample random input and next-token label batches.
        logits = model(x)  # Run a forward pass through the Transformer.
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))  # Compute next-token loss.
        optimizer.zero_grad()  # Clear gradients from the prior update.
        loss.backward()  # Backpropagate through attention and the rest of the network.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Limit large gradients.
        optimizer.step()  # Update weights using AdamW.
        if step == 1 or step % 20 == 0:  # Print occasional training status.
            print(f"Step {step:3d}/{train_steps} | loss = {loss.item():.4f}")  # Display loss.
    prompt = "attention "  # Set the text prefix for generation.
    prompt_ids = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long)  # Encode prompt IDs.
    output_ids = model.generate(prompt_ids.to(device), new_tokens=100)[0].tolist()  # Generate output.
    generated = "".join(itos[token] for token in output_ids)  # Convert token IDs back to text.
    print("\nGenerated text:\n" + generated)  # Display the model's sample.
    torch.save(model.state_dict(), "tiny_transformer_cpu.pt")  # Save learned network weights.
    print("\nSaved weights to tiny_transformer_cpu.pt")  # Confirm checkpoint path.


if __name__ == "__main__":  # Run training only when this file is executed directly.
    main()  # Start the example.
