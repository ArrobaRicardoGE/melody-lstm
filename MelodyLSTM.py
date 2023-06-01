import torch

class MelodyLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, device):
        super(MelodyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                  num_layers=num_layers, batch_first=True)
        
        self.fc_1 = torch.nn.Linear(hidden_size, 256)
        self.fc = torch.nn.Linear(256, output_size)
        
        self.relu = torch.nn.ReLU()
        self.device = device
        
    def forward(self, x):
        x.to(self.device)
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(self.device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, requires_grad=True).to(self.device)
        
        # output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn, _ = self.lstm(x)
        
        #hn = hn.view(-1, self.hidden_size)
        hn = hn[:, -1, :]
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        
        return out