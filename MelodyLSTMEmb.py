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
        
        '''
        x_in = torch.zeros([x.size(0), x.size(1), x.size(2) - 2 + 2 * self.embedding_dim], requires_grad=True)
        x_in[:, :, :x.size(2) - 2] = x[:, :, :x.size(2) - 2]
        x_in[:, :, x.size(2) - 2 : x.size(2) - 2 + self.embedding_dim] = self.embedding(x[:, :, -2].long())
        x_in[:, :, x.size(2) - 2 + self.embedding_dim : ] = self.embedding(x[:, :, -1].long())
        x_in = x_in.to(self.device)
        '''
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
        
        return out
        '''
        return torch.where(out < self.threshold, 
                           torch.zeros_like(out, requires_grad=True), 
                           torch.ones_like(out, requires_grad=True))
        '''