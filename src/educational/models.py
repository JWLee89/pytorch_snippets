import torch
import torch.nn.functional as F
from torch import nn


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(10, 20)
        self.fc_2 = nn.Linear(20, 30)
        self.fc_3 = nn.Linear(30, 2)
        self.relu = lambda x: F.relu(x)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        first_output = self.fc_1(input_tensor)
        second_output = self.fc_2(first_output)
        third_output = self.fc_3(second_output)
        final_output = self.relu(third_output)
        return final_output
