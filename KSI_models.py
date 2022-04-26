import torch
import torch.nn.functional as F
import torch.nn as nn


class KSI(nn.Module):
    def __init__(self, n_ksi_embedding, n_vocab):
        super().__init__()
        self.ksi_embedding = nn.Linear(n_vocab, n_ksi_embedding)
        self.ksi_attention = nn.Linear(n_ksi_embedding, n_ksi_embedding)
        self.ksi_output = nn.Linear(n_ksi_embedding, 1)
        
    def forward_ksi(self, notevec, wikivec):
        with torch.profiler.record_function("KSI Forward"):
            n = notevec.shape[0]
            n_codes = wikivec.shape[0]
            notevec = notevec.unsqueeze(1).expand(n, n_codes, -1)
            wikivec = wikivec.unsqueeze(0)
        
            z = torch.mul(wikivec, notevec)
            e = self.ksi_embedding(z)
            attention_scores = torch.sigmoid(self.ksi_attention(e))
            v = torch.mul(attention_scores, e)
            s = self.ksi_output(v)
            o = s.squeeze(2)
        
        return o
    
    
class ModifiedKSI(nn.Module):
    """
    Use weighted sum of intersecting elements of note and wiki vectors, to allow for 
    incorporating frequency information.
    """
    def __init__(self, n_ksi_embedding, n_vocab):
        super().__init__()
        self.weights = nn.Linear(2, 1, bias=False)
        self.ksi_embedding = nn.Linear(n_vocab, n_ksi_embedding)
        self.ksi_attention = nn.Linear(n_ksi_embedding, n_ksi_embedding)
        self.ksi_output = nn.Linear(n_ksi_embedding, 1)
        
    def forward_ksi(self, notevec, wikivec):
        with torch.profiler.record_function("Modified KSI Forward"):
            n = notevec.shape[0]
            n_codes = wikivec.shape[0]
            notevec = notevec.unsqueeze(1).expand(n, n_codes, -1)
            wikivec = wikivec.unsqueeze(0).expand(n, n_codes, -1)
            
            device = next(self.parameters()).device
        
            mask = torch.mul(torch.where(wikivec > 1, 
                                         torch.tensor(1, dtype=wikivec.dtype).to(device), 
                                         wikivec), 
                             torch.where(notevec > 1, 
                                         torch.tensor(1, dtype=notevec.dtype).to(device), 
                                         notevec))
        
            z = self.weights(torch.stack(
                [torch.where(mask > 0, notevec, 
                             torch.tensor(0, dtype=notevec.dtype).to(device)), 
                 torch.where(mask > 0, wikivec, 
                             torch.tensor(0, dtype=wikivec.dtype).to(device))], 
                dim=-1)).squeeze()
            e = self.ksi_embedding(z)
            attention_scores = torch.sigmoid(self.ksi_attention(e))
            v = torch.mul(attention_scores, e)
            s = self.ksi_output(v)
            o = s.squeeze(2)
        
        return o


class CNN(nn.Module):
    def __init__(self, n_words, n_wiki, n_embedding, ksi=None, **kwargs):
        super().__init__(**kwargs)
        self.ksi = ksi
        self.word_embeddings = nn.Embedding(n_words+1, n_embedding)
        self.dropout_embedding = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv1d(n_embedding, 100, 3)
        self.conv2 = nn.Conv1d(n_embedding, 100, 4)
        self.conv3 = nn.Conv1d(n_embedding, 100, 5)
        self.output = nn.Linear(n_embedding*3, n_wiki)
    
    def forward(self, note, notevec=None, wikivec=None):
        # batch_size, n = note.shape
        with torch.profiler.record_function("CNN Embedding"):
            embeddings = self.word_embeddings(note) # (batch_size, n, n_embedding)
            embeddings = self.dropout_embedding(embeddings)
            embeddings = embeddings.permute(0, 2, 1) # (batch_size, n_embedding, n)
        
        with torch.profiler.record_function("CNN Forward"):
            a1 = F.relu(self.conv1(embeddings))
            a1 = F.max_pool1d(a1, a1.shape[2])
            a2 = F.relu(self.conv2(embeddings))
            a2 = F.max_pool1d(a2, a2.shape[2])
            a3 = F.relu(self.conv3(embeddings))
            a3 = F.max_pool1d(a3, a3.shape[2])
            combined = torch.cat([a1, a2, a3], 1).squeeze(2)
       
            out = self.output(combined)
        if self.ksi:
            out += self.ksi.forward_ksi(notevec, wikivec)
        
        scores = torch.sigmoid(out)
        return scores


class CAML(nn.Module):
    def __init__(self, n_words, n_wiki, n_embedding, n_hidden=300, ksi=None, **kwargs):
        super().__init__(**kwargs)
        self.ksi = ksi
        self.word_embeddings = nn.Embedding(n_words+1, n_embedding)
        self.dropout_embedding = nn.Dropout(p=0.2)
        
        self.conv = nn.Conv1d(n_embedding, n_hidden, 10, padding=5)
        self.H = nn.Linear(n_hidden, n_wiki, bias=False)
        self.output = nn.Linear(n_hidden, n_wiki)
    
    def forward(self, note, notevec=None, wikivec=None):
        # batch_size, n = note.shape
        with torch.profiler.record_function("CAML Embedding"):
            embeddings = self.word_embeddings(note) # (batch_size, n, n_embedding)
            embeddings = self.dropout_embedding(embeddings)
            embeddings = embeddings.permute(0, 2, 1) # (batch_size, n_embedding, n)
        
        with torch.profiler.record_function("CAML Forward"):
            a1 = F.relu(self.conv(embeddings).permute(0, 2, 1))
            alpha = self.H.weight.matmul(a1.permute(0, 2, 1))
            alpha = F.softmax(alpha, dim=2)
            m = alpha.matmul(a1)
            out = self.output.weight.mul(m).sum(dim=2).add(self.output.bias)
            
        if self.ksi:
            out += self.ksi.forward_ksi(notevec, wikivec)
        
        scores = torch.sigmoid(out)
        return scores


class LSTM(nn.Module):
    def __init__(self, 
                 n_words, 
                 n_wiki, 
                 n_embedding,
                 n_hidden=100, 
                 batch_size=32, 
                 ksi=None, 
                 **kwargs):
        super().__init__(**kwargs)
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        
        self.ksi = ksi
        self.word_embeddings = nn.Embedding(n_words+1, n_embedding)
        self.dropout_embedding = nn.Dropout(p=0.2)
        
        self.lstm = nn.LSTM(n_embedding, n_hidden)
        self.hidden2code = nn.Linear(n_hidden, n_wiki)
        self.hidden = self.init_hidden()
        
    def init_hidden(self, batch_size=None, device='cpu'):
        if batch_size is None:
            batch_size = self.batch_size
        return (torch.zeros(1, batch_size, self.n_hidden).to(device),
                torch.zeros(1, batch_size, self.n_hidden).to(device))
    
    def forward(self, note, notevec=None, wikivec=None):
        device = next(self.parameters()).device
        if device != 'cpu' and self.hidden[0].device != device:
            self.hidden = tuple(h.to(device) for h in self.hidden)
        
        if note.shape[0] != self.batch_size:
            self.hidden = self.init_hidden(note.shape[0], device)
        
        # batch_size, n = note.shape
        with torch.profiler.record_function("LSTM Embedding"):
            embeddings = self.word_embeddings(note).permute(1, 0, 2) # (n, batch_size, n_embedding)
            embeddings = self.dropout_embedding(embeddings)
        
        with torch.profiler.record_function("LSTM Forward"):
            lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
            lstm_out = lstm_out.permute(1, 2, 0) # (batch_size, n_hidden, n)
            out = nn.MaxPool1d(lstm_out.shape[2])(lstm_out).squeeze(2)
            out = self.hidden2code(out)
            
        if self.ksi:
            out += self.ksi.forward_ksi(notevec, wikivec)
        
        scores = torch.sigmoid(out)
        return scores
