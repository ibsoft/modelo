import json
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import spacy
import matplotlib.pyplot as plt
from torchsummary import summary

# Ensure deterministic behavior
torch.manual_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Clear screen based on the platform


def clear_screen():
    if os.name == 'posix':  # For Linux and macOS
        os.system('clear')
    elif os.name == 'nt':   # For Windows
        os.system('cls')
    else:
        # For other operating systems, print a bunch of newlines to mimic clearing
        print('\n' * 100)


clear_screen()


def print_banner():
    banner = """
    
░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓█▓▒░            ░▒▓████████▓▒░▒▓████████▓▒░░▒▓██████▓▒░░▒▓██████████████▓▒░  
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░               ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░               ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░               ░▒▓█▓▒░   ░▒▓██████▓▒░ ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░               ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░               ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░               ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░ 
                                                                                                          
         
    """
    print(banner)


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
if len(data) >= 1000:
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


def check_model_architecture(encoder, decoder):
    # Encoder and Decoder Consistency
    print("Encoder and Decoder Consistency: Ensure that the output dimension of the encoder matches the input dimension of the decoder.")
    if encoder.embedding.num_embeddings == decoder.embedding.num_embeddings:
        print("Output dimension of the encoder matches the input dimension of the decoder.")
    else:
        print("Error: Output dimension of the encoder does not match the input dimension of the decoder.")

    # Encoder-Decoder Compatibility
    print("Encoder-Decoder Compatibility: Verify that the hidden dimensions of the encoder and decoder are compatible.")
    if encoder.rnn.hidden_size == decoder.rnn.hidden_size:
        print("Hidden dimensions of the encoder and decoder are compatible.")
    else:
        print("Error: Hidden dimensions of the encoder and decoder are not compatible.")

    # Encoder and Decoder Layers
    print("Encoder and Decoder Layers: Check if the number of layers in the encoder and decoder is suitable.")
    print("Encoder Layers:", encoder.rnn.num_layers)
    print("Decoder Layers:", decoder.rnn.num_layers)

# Convert text data to tensors


def text_to_tensor(text, max_length):
    tokens = tokenize(text)
    token_ids = [word2idx.get(token, word2idx['<unk>']) for token in tokens]
    token_ids = token_ids[:max_length-2]  # truncate if longer than max length
    token_ids = [word2idx['<sos>']] + token_ids + \
        [word2idx['<eos>']]  # add SOS and EOS tokens
    padding_length = max_length - len(token_ids)
    token_ids += [word2idx['<pad>']] * padding_length
    return torch.tensor(token_ids, dtype=torch.long)

# Define dataset class
# Define dataset class


class MyDataset(Dataset):
    def __init__(self, data, max_length):
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Assuming 'request' contains input sequence
        request_sequence = sample['request']
        # Assuming 'response' contains target sequence
        response_sequence = sample['response']

        # Tokenize input and target sequences
        input_tokens = tokenize(request_sequence)
        target_tokens = tokenize(response_sequence)

        # Add padding tokens
        # Subtract 1 for <eos> token
        input_tokens = input_tokens[:self.max_length - 1]
        # Subtract 1 for <eos> token
        target_tokens = target_tokens[:self.max_length - 1]
        input_tokens += ['<pad>'] * (self.max_length - len(input_tokens))
        target_tokens += ['<pad>'] * (self.max_length - len(target_tokens))

        # Convert tokens to tensor indices
        input_ids = [word2idx.get(token, word2idx['<unk>'])
                     for token in input_tokens]
        target_ids = [word2idx.get(token, word2idx['<unk>'])
                      for token in target_tokens]

        # Convert to tensors
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)

        return input_tensor, target_tensor


# Define DataLoader
BATCH_SIZE = 32
MAX_LENGTH = 50

train_dataset = MyDataset(train_data, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


train_dataset = MyDataset(train_data, MAX_LENGTH)
val_dataset = MyDataset(val_data, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define Decoder


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

    def init_hidden(self, batch_size):
        return torch.zeros(self.decoder.n_layers, batch_size, self.decoder.dec_hid_dim, device=self.device)

# Define function to plot training and validation loss


def plot_loss(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

# Training Loop


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for src, trg in tqdm(iterator, desc="Training Batches", total=len(iterator)):
        optimizer.zero_grad()

        output = model(src, trg)

        # Reshape output to [trg_len, batch_size, output_dim]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for src, trg in tqdm(iterator, desc="Evaluation Batches"):
            output = model(src, trg, 0)  # Turn off teacher forcing

            # Calculate loss
            # Ignore <sos> token
            loss = criterion(
                output[1:].view(-1, output.shape[-1]), trg[1:].view(-1))

            # Calculate mask to filter out padding tokens
            mask = (trg != word2idx['<pad>']).float()

            # Apply mask to ignore padding tokens in loss calculation
            masked_loss = (loss * mask.view(-1)).sum()

            # Accumulate loss
            epoch_loss += masked_loss.item()

    return epoch_loss / len(iterator)


N_EPOCHS = 1
CLIP = 1
patience = 3  # Adjust as needed
early_stop_count = 0
best_valid_loss = float('inf')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(len(word2idx), 256, 512, 2, dropout=0.5)
decoder = Decoder(len(word2idx), 256, 512, 2, dropout=0.5)
model = Seq2Seq(encoder, decoder, device).to(device)

# Now you can call the function like this:
check_model_architecture(encoder, decoder)


learning_rate = 0.0005
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])

train_losses = []
val_losses = []

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_loader, criterion)
    train_losses.append(train_loss)
    val_losses.append(valid_loss)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut3-model.pt')
        # Adjust input size as needed

        early_stop_count = 0  # Reset the count if validation loss improves
    else:
        # early_stop_count += 1
        early_stop_count = 0

    print(
        f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    if early_stop_count >= patience:
        print("Validation loss has been increasing for too long. Stopping training.")
        break

# Plot training and validation loss
plot_loss(train_losses, val_losses, 'loss_plot.png')

# Testing
print("Starting Testing...")
# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Adjust architecture to match the loaded model
encoder = Encoder(len(word2idx), 256, 512, 2, dropout=0.5)
# Adjust architecture to match the loaded model
decoder = Decoder(len(word2idx), 256, 512, 2, dropout=0.5)
model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(torch.load('tut3-model.pt'))
model.eval()


test_sentences = [
    "What is question 1?",
    "What is question 2?"

]


# Print tokenized and padded versions of a training and a test sentence
# Assuming the first training data is used
train_sentence = train_data[0]['request']
test_sentence = test_sentences[0]  # Assuming test_sentences is defined

# Tokenization
train_tokens = tokenize(train_sentence)
test_tokens = tokenize(test_sentence)
print("Tokenized Training Sentence:", train_tokens)
print("Tokenized Test Sentence:", test_tokens)

# Padding
train_tensor = text_to_tensor(train_sentence, MAX_LENGTH)
test_tensor = text_to_tensor(test_sentence, MAX_LENGTH)
print("Padded Training Tensor:", train_tensor)
print("Padded Test Tensor:", test_tensor)

# Prepare test data (example)
test_data = [text_to_tensor(sentence, MAX_LENGTH)
             for sentence in test_sentences]
# Since it's just a few sentences, we use batch_size=1
test_loader = DataLoader(test_data, batch_size=1)


# Inspect DataLoader Output
for batch in test_loader:
    print("Test Batch:", batch)


# Evaluate the model
predictions = []
with torch.no_grad():
    for src in tqdm(test_loader, desc="Testing Batches"):
        src = src.to(device)
        # Initialize hidden and cell states with batch size 1
        hidden = model.init_hidden(1)
        output = model(src, torch.zeros(
            src.shape[0], MAX_LENGTH).to(device), 0)
        output_dim = output.shape[-1]
        output = output.squeeze(1).argmax(dim=-1)  # Convert logits to indices

        # Print out the model's output
        print("Model Output:", output)

        predictions.append(output.cpu().numpy())

# Flatten the predictions list
flattened_predictions = [item for sublist in predictions for item in sublist]


# Convert indices back to words
predicted_sentences = []
for prediction in flattened_predictions:
    words = [idx2word[idx] for idx in prediction]
    predicted_sentence = ' '.join(words)
    predicted_sentences.append(predicted_sentence)

# Print the predicted sentences
for i, sent in enumerate(predicted_sentences, 1):
    print(f"Predicted Sentence {i}:", sent)

summary(model, input_size=[(32, 50), (32, 50)])
