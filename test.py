import json
import random
import torch
import torch.nn as nn
import spacy
from collections import Counter

# Build vocabulary
spacy_en = spacy.load('en_core_web_sm')

# Extract model parameters


def load_config(config_path):
    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: JSON decoding failed in '{config_path}'. Details: {e}")
        return None


config = load_config('config.json')
if config is not None:
    print("Configuration loaded successfully.")
else:
    print("Failed to load configuration.")

# Extract hyperparameters
training_params = config['training_params']
model_params = config['model_params']

# Extract training parameters
N_EPOCHS = training_params['N_EPOCHS']
CLIP = training_params['CLIP']
learning_rate = training_params['learning_rate']
patience = training_params['patience']

# Load data from JSON
try:
    with open('generated_data.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: JSON file not found. Please ensure that the file path is correct.")
    data = []
except json.JSONDecodeError:
    print("Error: JSON decoding failed. Please check if the file contains valid JSON data.")
    data = []


# Split data into training and validation sets
if len(data) >= 10:
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print("Train data size:", len(train_data))
    print("Val data size:", len(val_data))
else:
    print("Error: Dataset size is less than 1000 records. Please ensure that the dataset contains enough data.")
    train_data = []
    val_data = []

print("Dataset loaded successfully.")

# Tokenization function


def tokenize(text):
    return [token.text.lower() for token in spacy_en.tokenizer(text)]


# Build vocabulary
spacy_en = spacy.load('en_core_web_sm')
counter = Counter()
for d in train_data:
    counter.update(tokenize(d['request']))
    counter.update(tokenize(d['response']))
vocab = [word for word, freq in counter.items() if freq >= 2]

# Create word-to-index and index-to-word mappings
word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
for idx, word in enumerate(vocab, len(word2idx)):
    word2idx[word] = idx
idx2word = {idx: word for word, idx in word2idx.items()}


# Create word-to-index and index-to-word mappings
word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
for idx, word in enumerate(vocab, len(word2idx)):
    word2idx[word] = idx
idx2word = {idx: word for word, idx in word2idx.items()}

# Combine mappings into a dictionary
word_mappings = {'word2idx': word2idx, 'idx2word': idx2word}

# Save word_mappings to a JSON file
with open('word_mappings.json', 'w') as f:
    json.dump(word_mappings, f)

print("Word mappings saved to 'word_mappings.json'.")

# Load word-to-index and index-to-word mappings
with open('word_mappings.json', 'r') as f:
    word_mappings = json.load(f)
word2idx = word_mappings['word2idx']
idx2word = word_mappings['idx2word']


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, n_layers, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden  # Return only outputs and hidden, not cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, dec_hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, dec_hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.dec_hid_dim = dec_hid_dim

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.n_layers, batch_size, self.dec_hid_dim, device=device)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size,
                              trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1

        return outputs


# Load the trained model
encoder = Encoder(input_dim=len(word2idx), emb_dim=256,
                  enc_hid_dim=512, n_layers=2, dropout=0.5)
decoder = Decoder(output_dim=len(word2idx), emb_dim=256,
                  dec_hid_dim=512, n_layers=2, dropout=0.5)

model = Seq2Seq(encoder, decoder, device=torch.device('cpu'))

model.load_state_dict(torch.load(
    'tut3-model.pt', map_location=torch.device('cpu')))
model.eval()

# Define function to translate a sentence


def translate_sentence(sentence, model, device, max_len=50):
    model.eval()
    tokenized = [token.text.lower() for token in spacy_en.tokenizer(sentence)]
    tokenized = ['<sos>'] + tokenized + ['<eos>']
    numericalized = [word2idx.get(token, word2idx['<unk>'])
                     for token in tokenized]
    input_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(input_tensor)

    outputs = []
    previous_word = torch.tensor([word2idx['<sos>']], dtype=torch.long).to(
        device)  # Initialize previous_word
    for _ in range(max_len):
        with torch.no_grad():
            output, hidden = model.decoder(previous_word, hidden)
        best_guess = output.argmax(1).item()
        if best_guess == word2idx['<eos>']:
            break
        outputs.append(best_guess)
        previous_word = torch.tensor([best_guess], dtype=torch.long).to(device)

    translated_sentence = [idx2word[idx] for idx in outputs]
    return translated_sentence


# Test the model
test_sentence = "Find youtube videos of cricket batting drills"
translated_sentence = translate_sentence(
    test_sentence, model, device=torch.device('cpu'), max_len=50)
print(f"Input: {test_sentence}")
print(f"Prediction: {' '.join(translated_sentence)}")
