import typing as t
from collections import OrderedDict

import torch
from torch import nn

from src.educational.models import LinearModel


class ModuleNotFoundError(ValueError):
    pass


class HookManager:
    def __init__(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f'Must pass in nn.Module to "HookedModule". Found: {type(model)}')

        self.model = model
        self._forward_hooks = OrderedDict()
        # TODO: Later add some for backward hooks

    def get_module_by_name(self, target_module_name: str) -> nn.Module:
        """Find the target nn.module by name

        Args:
            target_module_name (str): The name of the target module we are looking for

        Raises:
            ModuleNotFoundError: Occurs when module with given name cannot be found in wrapped module

        Returns:
            nn.Module: The name of
        """
        for name, module in self.model.named_modules():
            if name == target_module_name:
                return module

        available_module_names = ', '.join([module_name for module_name, module in self.model.named_modules()])
        raise ModuleNotFoundError(
            f'Module with name: {target_module_name} not found. ' f'Available module names: {available_module_names}'
        )

    def add_forward_hook(self, module_name: str, forward_hook_function: t.Callable) -> None:
        """Given a target module name and forward hook function,
        add forward hook to the module with the given name

        Args:
            module_name (str): The name of the module name
            forward_hook_function (t.Callable): The forward hook function to call (for debugging purposes)

        Raises:
            TypeError: Occurs when foward_hook_function is not callable.
        """
        if not isinstance(forward_hook_function, t.Callable):
            raise TypeError('forward_hook_function must be callable. ' f'Passed in: {forward_hook_function}')
        # Need to de-register existing hook function
        self.remove_forward_hook(module_name)

        # Retrieve module
        module: nn.Module = self.get_module_by_name(module_name)

        # Register hook
        self._forward_hooks[module_name] = module.register_forward_hook(forward_hook_function)

    def remove_forward_hook(self, module_name: str) -> None:
        """_summary_

        Args:
            module_name (str): _description_

        Raises:
            TypeError: Occurs when the forward hook function is manipulated directly by a user,
            resulting in the function not being callable.
        """
        if module_name in self._forward_hooks:
            stored_forward_hook_function = self._forward_hooks[module_name]
            if not isinstance(stored_forward_hook_function, torch.utils.hooks.RemovableHandle):
                raise TypeError(
                    f'ERROR!! forward_hook_function handle should be stored '
                    f'instead of: {stored_forward_hook_function} of type: {type(stored_forward_hook_function)}. '
                    'Perhaps self._forward_hooks was overriden directly.'
                )
            # Clean-up
            stored_forward_hook_function.remove()
            del self._forward_hooks[module_name]


if __name__ == '__main__':
    model = LinearModel()
    hook_manager = HookManager(model)

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

    hook_manager.add_forward_hook('fc_1', forward_hook_function)

    # Try calling foward
    batch_size: int = 2
    input_shape: int = 10
    random_tensor = torch.randn((batch_size, input_shape))

    # Forward
    output = model(random_tensor)
    print(f'Output: {output}')

    # remove hook
    print('Removing hook!!!!!')
    print('-----------------------')
    hook_manager.remove_forward_hook('fc_1')
    output = model(random_tensor)
    print(f'Output: {output}')
