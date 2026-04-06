
import sys
import os
import streamlit as st
import torch

# add ROOT directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suggesion import model, word_to_idx, idx_to_word, seq_length, generate_next_words, LSTMPredictor


# sentence = "আমি বাংলায় গান গাই"

# result = generate_next_words(
#     model=model,
#     sentence=sentence,
#     word_to_idx=word_to_idx,
#     idx_to_word=idx_to_word,
#     seq_length=seq_length,
#     num_words=3
# )

# # print("Generated:", result)





# import streamlit as st

# st.title("Bengali Smart Keyboard")

# text = st.text_input("Type here")

# if text:
#     suggestions = generate_next_words(model, text, word_to_idx, idx_to_word, seq_length)
#     st.write("Suggestions:", suggestions)




@st.cache_resource
def load_model():
    checkpoint = torch.load("lstm_bengali_full.pth", map_location="cpu")

    model = LSTMPredictor(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=100,
        hidden_size=512
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


model, checkpoint = load_model()

word_to_idx = checkpoint["word_to_idx"]
idx_to_word = checkpoint["idx_to_word"]
seq_length = checkpoint["seq_length"]


# ---------------- SUGGESTION FUNCTION ----------------
def get_suggestions(sentence, top_k=5):

    unk_idx = word_to_idx.get("UNK", 0)
    pad_idx = word_to_idx.get("PAD", 0)

    tokens = sentence.lower().split()
    tokens = tokens[-seq_length:]

    indices = [word_to_idx.get(t, unk_idx) for t in tokens]

    if len(indices) < seq_length:
        indices = [pad_idx] * (seq_length - len(indices)) + indices

    input_tensor = torch.tensor([indices])

    with torch.no_grad():
        output = model(input_tensor)

        probs = torch.softmax(output, dim=1)

        top_probs, top_indices = torch.topk(probs, k=top_k)

        suggestions = [
            idx_to_word[idx.item()] for idx in top_indices[0]
        ]

    return suggestions


# ---------------- UI ----------------
st.title("⌨️ Bengali Smart Keyboard (Live)")

text = st.text_input("Type here (live suggestions will update)")

# LIVE UPDATE SECTION
if text:
    suggestions = get_suggestions(text)

    st.markdown("### 🔮 Suggestions:")

    cols = st.columns(len(suggestions))

    for i, word in enumerate(suggestions):
        with cols[i]:
            if st.button(word):
                text = text + " " + word
                st.experimental_rerun()

# optional debug
st.write("Current text:", text)