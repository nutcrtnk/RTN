import torch
import torch.nn as nn
import torch.nn.functional as F


class UpdateGate(nn.Module):

    def __init__(self, input_size, hidden_size, forget_bias=-1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.fc_in = torch.nn.Linear(input_size, self.hidden_size, bias=False)
        self.fc_h0 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.act = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.fc_h0.bias, self.forget_bias)

    def forward(self, input, hidden):
        z = self.act(self.fc_h0(hidden) + self.fc_in(input))
        return z


class RTNRnn(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.update_gate = UpdateGate(input_size, input_size)

    def forward(self, input, hidden):
        outputs = []
        for step in range(input.size(1)):
            inp = input[:, step]
            z = self.update_gate(inp, hidden)
            hidden = (1 - z) * hidden + z * inp
            outputs.append(hidden.clone())
        outputs = torch.stack(outputs, 1)
        return outputs, hidden
