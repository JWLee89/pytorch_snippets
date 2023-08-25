import inspect
import typing as t
from enum import Enum

from torch import nn


class HookType(Enum):
    FORWARD_PRE_HOOK = 1
    FORWARD = 2
    BACKWARD = 3


def get_type_error_message(actual_value: t.Any, expected_type: t.Any) -> str:
    """Given an actual input and the expected type, give a common message
    that users can analyze to see what they are doing wrong

    Args:
        actual_value (t.Any): The actual value inserted by the user
        expected_type (t.Any): The expected type that the actual value should be

    Returns:
        str: A string error message for users to read on the console for debugging applications.
    """
    if inspect.isclass(actual_value):
        actual_data_type = actual_value
    else:
        actual_data_type = type(actual_value)
    return f'Input must be of type: {expected_type}. Passed in: "{actual_value}" of type: {actual_data_type}'


class PyTorchHook:
    """
    A wrapper for existing PyTorch modules to add hooks
    """

    def __init__(self, module: nn.Module):
        if not isinstance(module, nn.Module):
            raise TypeError(get_type_error_message(module, nn.Module))

        self._forward_pre_hooks: t.List = []

    def register_hook(self, hook_type: HookType, hook_fn: t.Callable) -> None:
        """_summary_

        Args:
            hook_type (HookType): _description_
            hook_fn (t.Callable): _description_

        Raises:
            TypeError: _description_
            ValueError: _description_
        """
        for user_input, expected_type in zip((hook_type, hook_fn), (HookType, t.Callable)):
            if not isinstance(user_input, expected_type):
                raise TypeError(get_type_error_message(user_input, expected_type))

        # This is fixed inside of the PyTorch API, so we can leave it as it is.
        # The probability of additional hook types being introduced in PyTorch is very low.
        if hook_type == HookType.FORWARD_PRE_HOOK:
            print('forward pre hook')

        elif hook_type == HookType.FORWARD:
            print('forward')

        elif hook_type == HookType.BACKWARD:
            print('backward')

        else:
            raise ValueError(f'Invalid hook_type: {hook_type}. Choose from: {HookType._member_names_}')
        print(hook_type)


class Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        return None


if __name__ == '__main__':
    m = PyTorchHook(Module())
    m.register_hook(HookType.FORWARD, print)
