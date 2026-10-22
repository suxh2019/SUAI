"""Small English-to-Chinese encoder-decoder Transformer, trained from scratch on CPU.

Educational demo: a tiny dataset enables memorization of simple phrases, NOT general translation.
"""
import random  # Control Python's random example selection.
import re  # Split English sentences into word tokens.
import torch  # Tensor operations and autograd.
from torch import nn  # Neural network building blocks.
import torch.nn.functional as F  # Cross-entropy loss and activations.

SEED = 42  # Fixed seed for repeatability.
random.seed(SEED)  # Seed Python random-number generator.
torch.manual_seed(SEED)  # Seed PyTorch random-number generator.
torch.set_num_threads(2)  # Avoid excessive CPU threading on laptops.
DEVICE = torch.device('cpu')  # Require no GPU.

# Small paired corpus; target Chinese is tokenized character-by-character.
PAIRS = [
    ('hello', '你好'), ('good morning', '早上好'), ('good night', '晚安'),
    ('thank you', '谢谢你'), ('you are welcome', '不客气'),
    ('i love you', '我爱你'), ('i am happy', '我很开心'),
    ('i am tired', '我累了'), ('i am hungry', '我饿了'),
    ('i am a student', '我是学生'), ('i like tea', '我喜欢茶'),
    ('i like coffee', '我喜欢咖啡'), ('i like apples', '我喜欢苹果'),
    ('i like cats', '我喜欢猫'), ('i like dogs', '我喜欢狗'),
    ('this is a book', '这是一本书'), ('this is my home', '这是我的家'),
    ('where are you', '你在哪里'), ('how are you', '你好吗'),
    ('what is your name', '你叫什么名字'), ('see you tomorrow', '明天见'),
    ('the sky is blue', '天空是蓝色的'), ('the sun is bright', '太阳很明亮'),
    ('the water is cold', '水很冷'), ('the food is good', '食物很好吃'),
    ('please sit down', '请坐下'), ('please help me', '请帮帮我'),
    ('open the door', '打开门'), ('close the door', '关上门'),
    ('have a nice day', '祝你今天愉快'),
]

# Define special IDs so padding and sequence boundaries have consistent meanings.
PAD, BOS, EOS, UNK = '<pad>', '<bos>', '<eos>', '<unk>'
SPECIAL = [PAD, BOS, EOS, UNK]  # Keep same special-token order in both vocabularies.

def english_tokens(sentence):  # Tokenize English by words and basic punctuation.
    return re.findall(r"[a-z]+|[^\w\s]", sentence.lower())  # Return a list of tokens.

def chinese_tokens(sentence):  # Tokenize Chinese one character at a time.
    return list(sentence.replace(' ', ''))  # Remove spaces; separate characters.

def make_vocab(sequences):  # Create a vocabulary from tokenized sentences.
    words = sorted({token for seq in sequences for token in seq})  # Gather unique tokens.
    itos = SPECIAL + words  # Index-to-token lookup list.
    stoi = {token: i for i, token in enumerate(itos)}  # Token-to-index lookup dict.
    return stoi, itos  # Return both mappings.

src_stoi, src_itos = make_vocab([english_tokens(en) for en, _ in PAIRS])  # English vocab.
tgt_stoi, tgt_itos = make_vocab([chinese_tokens(zh) for _, zh in PAIRS])  # Chinese vocab.
PAD_ID, BOS_ID, EOS_ID, UNK_ID = range(4)  # Special token integer IDs.

def encode(tokens, vocab):  # Convert a token sequence to integer IDs.
    return [BOS_ID] + [vocab.get(t, UNK_ID) for t in tokens] + [EOS_ID]  # Add boundaries.

def pad_batch(sequences):  # Pad a variable-length batch into a rectangular tensor.
    longest = max(map(len, sequences))  # Find longest sequence in this batch.
    rows = [s + [PAD_ID] * (longest - len(s)) for s in sequences]  # Pad every sequence.
    return torch.tensor(rows, dtype=torch.long, device=DEVICE)  # Convert to tensor.

# Prepare all training pairs once; mini-batches are sampled with replacement.
DATA = [(encode(english_tokens(en), src_stoi), encode(chinese_tokens(zh), tgt_stoi))
        for en, zh in PAIRS]  # Each target includes <bos> and <eos>.

class EncoderLayer(nn.Module):  # One Transformer encoder block.
    def __init__(self, dim, heads):  # Configure size of hidden vectors and attention heads.
        super().__init__()  # Initialize nn.Module internals.
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)  # Self-attention.
        self.norm1 = nn.LayerNorm(dim)  # Normalize after first residual connection.
        self.norm2 = nn.LayerNorm(dim)  # Normalize after second residual connection.
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))  # FFN.

    def forward(self, x, src_pad):  # Encoder consumes only source representations.
        attended, _ = self.attn(x, x, x, key_padding_mask=src_pad, need_weights=False)  # Q=K=V=x.
        x = self.norm1(x + attended)  # Add skip connection and normalize.
        x = self.norm2(x + self.ff(x))  # Apply feed-forward block and second residual.
        return x  # Return contextual source embeddings.

class DecoderLayer(nn.Module):  # One Transformer decoder block.
    def __init__(self, dim, heads):  # Initialize attention and feed-forward modules.
        super().__init__()  # Initialize parent class.
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)  # Decoder attention.
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)  # Encoder-decoder attention.
        self.norm1 = nn.LayerNorm(dim)  # Normalization after masked self-attention.
        self.norm2 = nn.LayerNorm(dim)  # Normalization after cross-attention.
        self.norm3 = nn.LayerNorm(dim)  # Normalization after feed-forward block.
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))  # FFN.

    def forward(self, y, memory, tgt_pad, src_pad, causal_mask):  # Decode given encoder memory.
        out, _ = self.self_attn(y, y, y, attn_mask=causal_mask,
                                     key_padding_mask=tgt_pad, need_weights=False)  # Attend to past targets.
        y = self.norm1(y + out)  # Residual connection and normalization.
        out, _ = self.cross_attn(y, memory, memory, key_padding_mask=src_pad,
                                      need_weights=False)  # Queries=decoder; keys/values=encoder.
        y = self.norm2(y + out)  # Mix source information through cross-attention.
        y = self.norm3(y + self.ff(y))  # Apply feed-forward network.
        return y  # Return updated decoder representations.

class TinyTranslator(nn.Module):  # Full encoder-decoder Transformer.
    def __init__(self, dim=48, heads=4, layers=2, max_len=64):  # Tiny CPU-friendly configuration.
        super().__init__()  # Initialize module registration.
        self.src_embed = nn.Embedding(len(src_itos), dim)  # English token embeddings.
        self.tgt_embed = nn.Embedding(len(tgt_itos), dim)  # Chinese token embeddings.
        self.pos_embed = nn.Embedding(max_len, dim)  # Learned positional embeddings.
        self.encoders = nn.ModuleList([EncoderLayer(dim, heads) for _ in range(layers)])  # Encoder stack.
        self.decoders = nn.ModuleList([DecoderLayer(dim, heads) for _ in range(layers)])  # Decoder stack.
        self.output = nn.Linear(dim, len(tgt_itos))  # Predict vocabulary logits at every position.

    def add_positions(self, ids, embedding):  # Combine token and position representations.
        positions = torch.arange(ids.size(1), device=ids.device)  # Positions 0, 1, ... T-1.
        return embedding(ids) + self.pos_embed(positions)[None, :, :]  # Broadcast across batch.

    def forward(self, src_ids, tgt_input_ids):  # Source and shifted target input IDs.
        src_pad = src_ids.eq(PAD_ID)  # True where English sequence has padding.
        tgt_pad = tgt_input_ids.eq(PAD_ID)  # True where decoder input has padding.
        x = self.add_positions(src_ids, self.src_embed)  # Encode English token positions.
        for layer in self.encoders:  # Pass through all encoder layers.
            x = layer(x, src_pad)  # Build contextual source representations.
        y = self.add_positions(tgt_input_ids, self.tgt_embed)  # Embed Chinese prefix.
        length = tgt_input_ids.size(1)  # Target sequence length.
        mask = torch.triu(torch.ones(length, length, device=DEVICE, dtype=torch.bool), diagonal=1)  # Hide future words.
        for layer in self.decoders:  # Pass through all decoder layers.
            y = layer(y, x, tgt_pad, src_pad, mask)  # Masked self-attention + cross-attention.
        return self.output(y)  # Output [batch, time, Chinese_vocab] logits.

model = TinyTranslator().to(DEVICE)  # Instantiate model on CPU.
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)  # Optimizer for all parameters.

@torch.no_grad()  # Disable gradient tracking for inference.
def translate(sentence, max_new_tokens=24):  # Generate Chinese autoregressively.
    model.eval()  # Put model into evaluation mode.
    src_ids = pad_batch([encode(english_tokens(sentence), src_stoi)])  # Encode English source.
    generated = [BOS_ID]  # Begin decoder prefix with special start token.
    for _ in range(max_new_tokens):  # Generate one Chinese token per iteration.
        tgt_ids = torch.tensor([generated], dtype=torch.long, device=DEVICE)  # Current prefix.
        logits = model(src_ids, tgt_ids)  # Run encoder and decoder.
        next_id = int(logits[0, -1].argmax())  # Greedy next-token prediction.
        if next_id == EOS_ID:  # Stop at end-of-sequence token.
            break  # End decoding loop.
        generated.append(next_id)  # Append predicted token for the next step.
    return ''.join(tgt_itos[i] for i in generated[1:] if i >= len(SPECIAL))  # Join Chinese chars.

def main():  # Start training and demonstrate translation.
    print(f'Examples: {len(PAIRS)} | EN vocab: {len(src_itos)} | ZH vocab: {len(tgt_itos)}')  # Summarize dataset.
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')  # Report model size.
    model.train()  # Enable training mode.
    steps = 300  # Number of gradient updates; raise this for a better fit.
    batch_size = 16  # Small batches are practical on a CPU.
    for step in range(1, steps + 1):  # Run training iterations.
        batch = random.choices(DATA, k=batch_size)  # Sample bilingual pairs.
        src = pad_batch([example[0] for example in batch])  # Batch English inputs.
        tgt = pad_batch([example[1] for example in batch])  # Batch Chinese targets.
        decoder_in = tgt[:, :-1]  # Teacher forcing: all target tokens except the last.
        target_out = tgt[:, 1:]  # Expected next tokens, shifted left by one.
        logits = model(src, decoder_in)  # Compute next-token predictions.
        loss = F.cross_entropy(logits.reshape(-1, len(tgt_itos)), target_out.reshape(-1),
                               ignore_index=PAD_ID)  # Exclude padding tokens from loss.
        optimizer.zero_grad()  # Clear stale accumulated gradients.
        loss.backward()  # Calculate gradients using backpropagation.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Bound gradient norm for stability.
        optimizer.step()  # Adjust trainable weights.
        if step % 50 == 0 or step == 1:  # Report learning progress periodically.
            print(f'Step {step:3d}/{steps} | loss = {loss.item():.4f}')  # Current batch loss.
    print('\nEnglish -> predicted Chinese (training-set examples):')  # Introduce demonstration.
    for en in ['hello', 'thank you', 'i like tea', 'good night', 'where are you']:
        print(f'{en:20s} -> {translate(en)}')  # Print model-generated translation.
    torch.save({'model': model.state_dict(), 'src_vocab': src_itos,
                'tgt_vocab': tgt_itos}, 'tiny_en_zh_transformer.pt')  # Save learned weights and vocab.
    print('\nSaved: tiny_en_zh_transformer.pt')  # Confirm output path.

if __name__ == '__main__':  # Only train when executed as a script.
    main()  # Begin experiment.
