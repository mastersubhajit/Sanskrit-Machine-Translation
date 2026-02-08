from flask import Flask, render_template, request
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import re
import pickle
from indicnlp.tokenize import indic_tokenize
torchtext.disable_torchtext_deprecation_warning()

from torchtext.data.utils import get_tokenizer

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab_transform.pkl")

# Load vocab_transform
with open(VOCAB_PATH, 'rb') as f:
    vocab_transform = pickle.load(f)

# Define special symbols and indices (must match A3.ipynb)
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

SRC_LANGUAGE = 'sa'
TRG_LANGUAGE = 'en'

PUNCT_TOKENS = {".", ",", "!", "?", ":", ";", "\"", "''", "``"}

# Normalization functions (must match A3.ipynb)
def normalize_sa(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_en(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

# Tokenization functions (must match A3.ipynb)
token_transform = {}
token_transform[SRC_LANGUAGE] = lambda x: indic_tokenize.trivial_tokenize(normalize_sa(x))
try:
    _spacy_en = get_tokenizer('spacy', language='en_core_web_sm')
    token_transform[TRG_LANGUAGE] = lambda x: _spacy_en(normalize_en(x))
except Exception:
    basic_en = get_tokenizer('basic_english')
    token_transform[TRG_LANGUAGE] = lambda x: basic_en(normalize_en(x))

# Text transform pipeline (must match A3.ipynb)
def sequential_transforms(*transforms):
    def func(txt):
        for transform in transforms:
            txt = transform(txt)
        return txt
    return func

def tensor_transform(token_ids):
    return torch.cat((
        torch.tensor([SOS_IDX]),
        torch.tensor(token_ids),
        torch.tensor([EOS_IDX])
    ))

text_transform = {}
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(
        token_transform[ln],
        vocab_transform[ln],
        tensor_transform
    )

def token_ids(tokens, vocab):
    stoi = vocab.get_stoi()
    return [stoi[t] for t in tokens if t in stoi]

PUNCT_IDS = set(token_ids(PUNCT_TOKENS, vocab_transform[TRG_LANGUAGE]))

GLOSSARY_RULES = [
    (r"\bit is not be\b", "it will never be"),
    (r"\bnot be not\b", "will never be"),
    (r"\bwhat is the the\b", "what is the"),
    (r"\bi will know the\b", "i will know"),
]

def postprocess_translation(text: str) -> str:
    cleaned = " ".join(text.split())
    for pattern, replacement in GLOSSARY_RULES:
        cleaned = re.sub(pattern, replacement, cleaned)
    return cleaned.strip()

# Model architecture (must match A3.ipynb)
device = torch.device("cpu")  # Flask app runs on CPU

INPUT_DIM = len(vocab_transform[SRC_LANGUAGE])
OUTPUT_DIM = len(vocab_transform[TRG_LANGUAGE])
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
DROPOUT = 0.5

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class GeneralAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)
        return F.softmax(energy, dim=1)

class AdditiveAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.W1 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.W2 = nn.Linear(hid_dim, hid_dim, bias=False)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        hidden = hidden.unsqueeze(1)
        energy = torch.tanh(self.W1(encoder_outputs) + self.W2(hidden))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(hid_dim + emb_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim + hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1)
        context = context.unsqueeze(0)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(
            torch.cat((output.squeeze(0), context.squeeze(0), embedded.squeeze(0)), dim=1)
        )
        return prediction, hidden, attn_weights.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(trg_len, batch_size, vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs

additive_ckpt = os.path.join(MODEL_DIR, "additive_best.pt")
general_ckpt = os.path.join(MODEL_DIR, "general_best.pt")
if os.path.exists(additive_ckpt):
    ckpt_path = additive_ckpt
elif os.path.exists(general_ckpt):
    ckpt_path = general_ckpt
else:
    raise FileNotFoundError("No checkpoint found in models/. Expected additive_best.pt or general_best.pt")

if os.path.basename(ckpt_path).startswith("additive"):
    attention = AdditiveAttention(HID_DIM)
else:
    attention = GeneralAttention()

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DROPOUT, attention)
model = Seq2Seq(enc, dec, device).to(device)

model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# Translation inference function
def _decode_step(output, step, min_len, min_content_len, repeat_penalty, banned_ids, prev_ids):
    if step < min_len:
        output[0, EOS_IDX] = float("-inf")
    if step < min_content_len:
        for pid in PUNCT_IDS.union(banned_ids):
            output[0, pid] = float("-inf")
    if repeat_penalty > 1.0:
        for pid in set(prev_ids):
            output[0, pid] /= repeat_penalty
    if prev_ids:
        output[0, prev_ids[-1]] = float("-inf")
    return output

def beam_search_translate(
    sentence,
    model,
    src_language,
    trg_language,
    max_len=50,
    min_len=2,
    min_content_len=2,
    repeat_penalty=1.2,
    banned_ids=None,
    beam_size=6,
    length_penalty=0.9,
):
    model.eval()
    if isinstance(sentence, str):
        tokens = token_transform[src_language](sentence)
    else:
        tokens = [token.lower() for token in sentence]

    src_indexes = [vocab_transform[src_language].get_stoi()["<sos>"]] + \
        [vocab_transform[src_language].get_stoi().get(token, UNK_IDX) for token in tokens] + \
        [vocab_transform[src_language].get_stoi()["<eos>"]]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    banned_ids = banned_ids or set()
    beams = [([SOS_IDX], hidden, 0.0)]
    completed = []

    for step in range(max_len):
        new_beams = []
        for seq, h, score in beams:
            if seq[-1] == EOS_IDX:
                completed.append((seq, score))
                continue

            trg_tensor = torch.LongTensor([seq[-1]]).to(device)
            with torch.no_grad():
                output, new_h, _ = model.decoder(trg_tensor, h, encoder_outputs)
            output = _decode_step(output, step, min_len, min_content_len, repeat_penalty, banned_ids, seq)
            log_probs = torch.log_softmax(output, dim=1).squeeze(0)
            topk = torch.topk(log_probs, k=beam_size)
            for idx, lp in zip(topk.indices.tolist(), topk.values.tolist()):
                new_beams.append((seq + [idx], new_h, score + lp))

        if not new_beams:
            break

        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_size]

    if completed:
        candidates = completed
    else:
        candidates = [(seq, score) for seq, _, score in beams]
    if not candidates:
        return "<unk>"
    def norm_score(seq, score):
        length = max(1, len(seq) - 1)
        return score / (length ** length_penalty)

    best_seq, best_score = max(candidates, key=lambda x: norm_score(x[0], x[1]))
    trg_tokens = [vocab_transform[trg_language].get_itos()[i] for i in best_seq]
    output_tokens = [t for t in trg_tokens[1:-1] if t not in ["<pad>"] and t.strip() != ""]
    if not output_tokens:
        output_tokens = ["<unk>"]
    return postprocess_translation(" ".join(output_tokens))

def translate_sentence_inference(
    sentence,
    model,
    src_language,
    trg_language,
    max_len=50,
):
    return beam_search_translate(
        sentence,
        model,
        src_language,
        trg_language,
        max_len=max_len,
    )

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    if request.method == 'POST':
        text_to_translate = request.form['text']
        translated_text = translate_sentence_inference(
            text_to_translate, model, SRC_LANGUAGE, TRG_LANGUAGE
        )
        return render_template('index.html', original_text=text_to_translate, translated_text=translated_text)

if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    app.run(port=8000, debug=True)