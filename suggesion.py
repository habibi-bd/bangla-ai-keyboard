import torch
import torch.nn as nn

# ---------------- MODEL ----------------
class LSTMPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embed)
        out = self.fc(hidden.squeeze(0))
        return out


# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- LOAD CHECKPOINT ----------------
checkpoint = torch.load("lstm_bengali_full.pth", map_location=device)

vocab_size = checkpoint["vocab_size"]
word_to_idx = checkpoint["word_to_idx"]
idx_to_word = checkpoint["idx_to_word"]
seq_length = checkpoint["seq_length"]


# ---------------- INIT MODEL ----------------
model = LSTMPredictor(
    vocab_size=vocab_size,
    embedding_dim=100,
    hidden_size=512
).to(device)


# IMPORTANT: load weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully!")


def generate_next_words(model, sentence, word_to_idx, idx_to_word, seq_length, num_words=3, temperature=1.0):
    model.eval()

    unk_idx = word_to_idx.get("UNK", 0)
    pad_idx = word_to_idx.get("PAD", 0)

    tokens = sentence.lower().split()
    generated = tokens.copy()

    with torch.no_grad():
        for _ in range(num_words):

            # take last seq_length tokens
            current_tokens = generated[-seq_length:]

            # convert to indices
            indices = [word_to_idx.get(tok, unk_idx) for tok in current_tokens]

            # padding
            if len(indices) < seq_length:
                indices = [pad_idx] * (seq_length - len(indices)) + indices

            input_tensor = torch.tensor([indices], dtype=torch.long).to(next(model.parameters()).device)

            # forward pass
            output = model(input_tensor)  # logits

            # temperature scaling
            logits = output / temperature

            probs = torch.softmax(logits, dim=1)

            # reduce UNK probability (important for Bengali)
            if "UNK" in word_to_idx:
                probs[0, word_to_idx["UNK"]] *= 0.3

            probs = probs / probs.sum(dim=1, keepdim=True)

            # sample next word
            predicted_idx = torch.multinomial(probs, num_samples=1).item()

            predicted_word = idx_to_word.get(predicted_idx, "<UNK>")

            generated.append(predicted_word)

    return " ".join(generated)

# sentence = "আমি বাংলায় গান গাই"

# result = generate_next_words(
#     model=model,
#     sentence=sentence,
#     word_to_idx=word_to_idx,
#     idx_to_word=idx_to_word,
#     seq_length=seq_length,
#     num_words=3)
# print("Generated:", result)