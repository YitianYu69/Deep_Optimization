"""Native represented-value maxima and exact relative routing oracle."""

from torch.nn import functional as F

from .windows import encode_locations


def forward(x, g, options, samples):
    arguments = dict(options)
    if samples is not None:
        arguments["_random_samples"] = samples.to(x.dtype)
    if g.family == "power":
        value = x.abs() if g.power > 0 else -x.abs()
        result, indices = getattr(F, f"max_pool{g.rank}d")(value, options["kernel_size"], options.get("stride"),
                                                           0, 1, options.get("ceil_mode", False), True)
        if g.power < 0:
            result = -result
    else:
        arguments["return_indices"] = True
        result, indices = g.spec.native(x, **arguments)
    codes = encode_locations(indices, g, samples)
    if g.family == "power":
        selected = x.reshape(g.batch * g.channels, -1).gather(1, indices.reshape(g.batch * g.channels, -1))
        signs = (selected > 0).long() + (selected < 0).long() * 2
        codes = codes * 3 + signs.reshape(g.output_shape)
    return result, indices, codes
