import typing as t

import torch
import torch.nn as nn

from src.educational.models import LinearModel

if __name__ == '__main__':
    batch_size: int = 2
    input_shape: int = 10
    random_tensor = torch.randn((batch_size, input_shape))

    model = LinearModel()

    def forward_hook_function(
        layer: nn.Module, input_tensor: t.Tuple[torch.Tensor, ...], output_tensor: t.Tuple[torch.Tensor, ...]
    ):
        """
        A hook function that is called after the forward function is processed

        Args:
            layer (nn.Module): The current layer that has registered forward hook
            input_tensor (t.Tuple[torch.Tensor, ...]): The input arguments (*args)
            output_tensor (t.Tuple[torch.Tensor, ...]): The output arguments (in most cases,
                                                        a tuple with a single output tensor)
        """

        print(f'Current layer: {layer}, input_tensor: {input_tensor}' f'output tensor: {output_tensor}')
        print('-' * 100)

    model.fc_1.register_forward_hook(forward_hook_function)
    model.fc_2.register_forward_hook(forward_hook_function)
    model.fc_3.register_forward_hook(forward_hook_function)

    output = model(random_tensor)
