import torch.nn as nn
import torch

def encode_text(text, word2idx):
    return [word2idx.get(word, word2idx["<UNK>"]) for word in text.split()]

def pad_sequence_to_length(seq, max_len, pad_idx):
    return seq + [pad_idx] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VanillaLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, output_size, attention_class = None, attn_dim = None, dropout = 0.3):
        super().__init__()

        self.hidden_size = hidden_size
        # Applying pretrained embeddings to tokens
        num_embeddings, embed_size = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = False)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, batch_first=True)
        # Dropout Layer
        self.dropout_layer = nn.Dropout(dropout)

        # Check if attention class is provided
        self.use_attention = attention_class is not None 
        if self.use_attention:
            # Instantiate Model
            self.attention = attention_class(self.hidden_size, self.hidden_size, attn_dim)
            
            # Final layer should be the concat of hn[-1] and the context vector
            self.fc = nn.Linear(self.hidden_size*2, output_size)
        else:
            self.fc = nn.Linear(self.hidden_size, output_size)




    def forward(self, x):
        x = self.embedding(x)
        x = x.to(device)
        # Set initial hidden state to 0
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)


        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout_layer(out)

        if self.use_attention:
            context, attn_weights = self.attention(out, hn[-1])
            context = context.squeeze(1)
            combined = torch.cat((context, hn[-1]), dim = 1)
            output = self.fc(combined)
            return output, attn_weights.squeeze(1)
        else:
            output = self.fc(hn[-1])
            return output


    