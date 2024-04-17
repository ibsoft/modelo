import torch
from model import Encoder, Decoder, Seq2Seq, tokenize, text_to_tensor, word2idx, idx2word
from torch.utils.data import DataLoader
import spacy
from tqdm import tqdm


MAX_LENGTH = 50

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(len(word2idx), 256, 512, 2, dropout=0.5)  # Adjust architecture to match the loaded model
decoder = Decoder(len(word2idx), 256, 512, 2, dropout=0.5)  # Adjust architecture to match the loaded model
model = Seq2Seq(encoder, decoder, device).to(device)
model.load_state_dict(torch.load('tut3-model.pt'))
model.eval()

# Define test sentences
test_sentences = [
    "What is question 1?",
    "What is question 2?"
]

# Tokenization and Padding
test_data = [text_to_tensor(sentence, MAX_LENGTH) for sentence in test_sentences]

# Prepare test DataLoader
test_loader = DataLoader(test_data, batch_size=1)  # Since it's just a few sentences, we use batch_size=1

# Evaluate the model
predictions = []
with torch.no_grad():
    for src in tqdm(test_loader, desc="Testing Batches"):
        src = src.to(device)
        hidden = model.init_hidden(1)  # Initialize hidden and cell states with batch size 1
        output = model(src, torch.zeros(src.shape[0], MAX_LENGTH).to(device), 0)
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
