"""Educational NLP pipeline: tokenization, vocabulary, neural sentiment model, evaluation.
Run: python nlp_sentiment_laptop.py  |  Dependency: pip install torch
"""
import random  # Randomly shuffle training examples in a reproducible way.
import re  # Tokenize text using a basic regular expression.
from collections import Counter  # Count vocabulary word frequencies.
import torch  # Work with tensors and gradients.
from torch import nn  # Build neural-network layers.
from torch.utils.data import DataLoader, TensorDataset  # Create minibatches.

SEED = 42  # Fixed seed for reproducible experiments.
random.seed(SEED)  # Seed Python's random-number generator.
torch.manual_seed(SEED)  # Seed PyTorch initialization and sampling.
torch.set_num_threads(1)  # Keep CPU work lightweight for laptops.
DEVICE = torch.device('cpu')  # Force the implementation to use the CPU.

# Each tuple contains a review and its label: 1=positive, 0=negative.
examples = [
    ('I love this amazing movie', 1), ('The film was wonderful and fun', 1),
    ('What a great and enjoyable story', 1), ('The acting was excellent', 1),
    ('This was a fantastic experience', 1), ('I really liked this film', 1),
    ('A beautiful and inspiring movie', 1), ('The ending was great', 1),
    ('I enjoyed every minute', 1), ('Brilliant acting and lovely music', 1),
    ('A delightful and entertaining story', 1), ('The characters were wonderful', 1),
    ('I recommend this excellent film', 1), ('Such a fun and uplifting movie', 1),
    ('The movie was charming', 1), ('I loved the beautiful ending', 1),
    ('I hate this terrible movie', 0), ('The film was boring and awful', 0),
    ('What a bad and disappointing story', 0), ('The acting was horrible', 0),
    ('This was a dreadful experience', 0), ('I really disliked this film', 0),
    ('An annoying and confusing movie', 0), ('The ending was bad', 0),
    ('I regretted every minute', 0), ('Terrible acting and irritating music', 0),
    ('A dull and frustrating story', 0), ('The characters were awful', 0),
    ('I cannot recommend this bad film', 0), ('Such a boring and depressing movie', 0),
    ('The movie was disappointing', 0), ('I hated the terrible ending', 0),
]

def tokenize(text):  # Convert raw text into a list of lowercase word tokens.
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())  # Retain letters and contractions.

positive = [row for row in examples if row[1] == 1]  # Separate positive reviews for stratification.
negative = [row for row in examples if row[1] == 0]  # Separate negative reviews for stratification.
random.shuffle(positive)  # Shuffle positive examples reproducibly.
random.shuffle(negative)  # Shuffle negative examples reproducibly.
train_examples = positive[:12] + negative[:12]  # Balanced training set of 24 reviews.
test_examples = positive[12:] + negative[12:]  # Balanced held-out set of 8 reviews.
random.shuffle(train_examples)  # Mix the training labels.
counts = Counter(word for text, _ in train_examples for word in tokenize(text))  # Count training words only.
vocab = {'<PAD>': 0, '<UNK>': 1}  # Reserve indices for padding and unknown words.
for word in sorted(counts):  # Add training vocabulary in a deterministic order.
    vocab[word] = len(vocab)  # Assign each new word a unique integer index.
MAX_LEN = 10  # Pad or truncate each review to 10 tokens.

def encode(text):  # Convert a review into a fixed-length sequence of word IDs.
    ids = [vocab.get(word, vocab['<UNK>']) for word in tokenize(text)][:MAX_LEN]  # Handle unknown words.
    return ids + [vocab['<PAD>']] * (MAX_LEN - len(ids))  # Add zero padding.

def make_dataset(rows):  # Build features and labels as tensors.
    x = torch.tensor([encode(text) for text, _ in rows], dtype=torch.long)  # Word indices.
    y = torch.tensor([label for _, label in rows], dtype=torch.float32)  # Binary labels.
    return TensorDataset(x, y)  # Bundle corresponding inputs and targets.

loader = DataLoader(make_dataset(train_examples), batch_size=8, shuffle=True)  # Training minibatches.
test_x, test_y = make_dataset(test_examples).tensors  # Held-out evaluation tensors.

class SentimentClassifier(nn.Module):  # Define a small embedding-based text classifier.
    def __init__(self, vocab_size):  # Create all trainable layers.
        super().__init__()  # Initialize the parent PyTorch class.
        self.embedding = nn.Embedding(vocab_size, 24, padding_idx=0)  # Learn 24-dimensional word vectors.
        self.classifier = nn.Sequential(  # Map averaged word vectors to a sentiment logit.
            nn.Linear(24, 32),  # First fully connected layer.
            nn.ReLU(),  # Introduce nonlinearity.
            nn.Linear(32, 1),  # Produce one binary-classification logit.
        )

    def forward(self, token_ids):  # Define how text is processed by the model.
        embedded = self.embedding(token_ids)  # Shape: [batch, tokens, 24].
        mask = (token_ids != 0).unsqueeze(-1)  # Ignore padding when averaging.
        sums = (embedded * mask).sum(dim=1)  # Sum nonpadding token vectors.
        lengths = mask.sum(dim=1).clamp(min=1)  # Count nonpadding tokens safely.
        pooled = sums / lengths  # Mean-pool word embeddings into a sentence vector.
        return self.classifier(pooled).squeeze(-1)  # Return one logit per review.

model = SentimentClassifier(len(vocab)).to(DEVICE)  # Initialize the classifier on CPU.
loss_fn = nn.BCEWithLogitsLoss()  # Stable binary cross-entropy applied to logits.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Optimize weights using Adam.

for epoch in range(1, 101):  # Train for 100 passes over the small dataset.
    model.train()  # Activate training mode.
    total_loss = 0.0  # Track average training loss.
    for inputs, labels in loader:  # Iterate over minibatches.
        logits = model(inputs.to(DEVICE))  # Forward pass: generate predicted logits.
        loss = loss_fn(logits, labels.to(DEVICE))  # Compare predictions with targets.
        optimizer.zero_grad()  # Clear previous gradients.
        loss.backward()  # Compute derivatives using backpropagation.
        optimizer.step()  # Update trainable parameters.
        total_loss += loss.item() * len(labels)  # Accumulate batch-weighted loss.
    if epoch == 1 or epoch % 20 == 0:  # Report training progress periodically.
        print(f'Epoch {epoch:3d} | training loss: {total_loss/len(train_examples):.4f}')  # Show average loss.

model.eval()  # Set the model to evaluation mode.
with torch.no_grad():  # Disable gradients for efficient inference.
    probs = torch.sigmoid(model(test_x.to(DEVICE)))  # Convert logits into probabilities.
    predicted = (probs >= 0.5).float()  # Classify probabilities using a 0.5 threshold.
    accuracy = (predicted.cpu() == test_y).float().mean().item()  # Compute held-out accuracy.
print(f'\nHeld-out accuracy: {accuracy:.1%} ({len(test_examples)} test reviews)')  # Print the result.
print('Note: this tiny dataset is instructional; test accuracy is not a reliable real-world benchmark.')  # Scope.

def predict_sentiment(text):  # Infer sentiment for an arbitrary input sentence.
    model.eval()  # Use deterministic evaluation behavior.
    with torch.no_grad():  # Avoid gradient computation.
        ids = torch.tensor([encode(text)], dtype=torch.long, device=DEVICE)  # Encode a single sentence.
        probability = torch.sigmoid(model(ids)).item()  # Obtain positive-class probability.
    label = 'positive' if probability >= 0.5 else 'negative'  # Choose a predicted label.
    return label, probability  # Return label and numerical probability.

for review in ['What a wonderful film', 'This movie was terrible', 'I enjoyed the acting']:
    label, confidence = predict_sentiment(review)  # Predict an example review.
    print(f'{review!r} -> {label} (P(positive)={confidence:.3f})')  # Show prediction.

torch.save({'model_state_dict': model.state_dict(), 'vocab': vocab, 'max_len': MAX_LEN}, 'nlp_sentiment_model.pt')  # Save learned parameters and vocabulary.
print('\nSaved model to nlp_sentiment_model.pt')  # Confirm where the checkpoint was written.
