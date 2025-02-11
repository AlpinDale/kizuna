import functools

import torch
from loguru import logger

try:
    import kizuna._C
except ImportError as e:
    logger.warning(f"Failed to import from kizuna._C with {e}")


def hint_on_error(fn):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AttributeError as e:
            msg = (
                f"Error in calling custom op {fn.__name__}: {e}\n"
                f"Possibly you have built or installed an obsolete version of kizuna.\n"
                f"Please try a clean build and install of kizuna,"
                f"or remove old built files such as kizuna/*.so and build/ ."
            )
            logger.error(msg)
            raise e

    return wrapper


# layer norm ops
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    torch.ops._C.rms_norm(out, input, weight, epsilon)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float) -> None:
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)

def layer_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
               bias: torch.Tensor, epsilon: float) -> None:
    torch.ops._C.layer_norm(out, input, weight, bias, epsilon)

def fused_add_layer_norm(input: torch.Tensor, residual: torch.Tensor,
                        weight: torch.Tensor, bias: torch.Tensor, epsilon: float) -> None:
    torch.ops._C.fused_add_layer_norm(input, residual, weight, bias, epsilon)

def ada_layer_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
                  bias: torch.Tensor, epsilon: float) -> None:
    torch.ops._C.ada_layer_norm(out, input, weight, bias, epsilon)


names_and_values = globals()
names_and_values_to_update = {}
# prepare variables to avoid dict size change during iteration
k, v, arg = None, None, None
fn_type = type(lambda x: x)
for k, v in names_and_values.items():
    # find functions that are defined in this file and have torch.Tensor
    # in their annotations. `arg == "torch.Tensor"` is used to handle
    # the case when users use `import __annotations__` to turn type
    # hints into strings.
    if isinstance(v, fn_type) \
        and v.__code__.co_filename == __file__ \
        and any(arg is torch.Tensor or arg == "torch.Tensor"
                for arg in v.__annotations__.values()):
        names_and_values_to_update[k] = hint_on_error(v)

names_and_values.update(names_and_values_to_update)
del names_and_values_to_update, names_and_values, v, k, fn_type
