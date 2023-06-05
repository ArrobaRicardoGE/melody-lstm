import torch

class MelodyLSTMEmb(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, 
                 device, num_embeddings=51, embedding_dim=10, threshold=0.2):
        super(MelodyLSTMEmb, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device
        self.embedding_dim = embedding_dim
        self.threshold = threshold
        
        self.embedding = torch.nn.Embedding(num_embeddings=num_embeddings, 
                                            embedding_dim=embedding_dim)
        
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                  num_layers=num_layers, batch_first=True)
        
        self.fc_1 = torch.nn.Linear(hidden_size, 256)
        self.fc = torch.nn.Linear(256, output_size)
        
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        # x -> (batch, seq, features)
        x.to(self.device)
        indices = x[:, :, -2:]
        x = x[:, :, :-2]
        x = torch.cat((x, self.embedding(indices[:, :, 0].long())), dim=2)
        x = torch.cat((x, self.embedding(indices[:, :, 1].long())), dim=2)
        
        hn, _ = self.lstm(x)
        
        hn = hn[:, -1, :]
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.sigmoid(out)
        out = self.fc(out)
        
        return torch.sigmoid(out)