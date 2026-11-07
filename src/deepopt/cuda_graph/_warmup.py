"""Side-stream execution mechanics; callers own warmup state and grad modes."""

import torch


def warmup_on_stream(function, args, kwargs, *, iterations, stream, device):
    """Run on an already validated stream and always join the caller's stream.

    This deliberately preserves function side effects. Capture callers decide
    whether inputs, model buffers or RNG state need restoration afterwards.
    """
    current = torch.cuda.current_stream(device)
    stream.wait_stream(current)
    try:
        with torch.cuda.stream(stream):
            for _ in range(iterations):
                function(*args, **kwargs)
    finally:
        current.wait_stream(stream)
