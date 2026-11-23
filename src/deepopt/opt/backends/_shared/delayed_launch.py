"""Host launch planning and bounded metadata reuse, independent of Triton."""

from collections import OrderedDict
from weakref import ref

import torch


CONVERSION_KEYS = ("M", "N", "SM", "SN", "NORMAL", "TRANSPOSE", "BIAS", "OBSERVE", "CHECK", "DEVICE")


def conversion_cache_key(x, scale, dtype, *, normal, transposed, bias, observe, check_finite):
    m, n = x.shape
    return ((m, n, *x.stride(), normal, transposed, bias, observe, check_finite, x.device.index)
            + tuple(map(str, (x.dtype, scale.dtype, dtype, dtype, torch.float32, torch.float32, torch.float32,
                              torch.int32 if check_finite else torch.float32))))


def scratch_tile(kernel, key):
    # Triton bypasses its key cache for a single configured candidate.
    config = kernel.configs[0] if len(kernel.configs) == 1 else kernel.cache.get(key)
    if config is not None:
        return config.kwargs["BM"], config.kwargs["BN"]
    return min(config.kwargs["BM"] for config in kernel.configs), min(config.kwargs["BN"] for config in kernel.configs)


def observation_rows(observations, block=1024):
    rows, programs = [], 0
    for group, value in observations:
        size = value.numel()
        if size < 1:
            raise ValueError("Amax observations must be nonempty")
        programs += (size + block - 1) // block
        rows.append((value.data_ptr(), size, group, programs))
    return rows, programs


def storage_key(value):
    storage = value.untyped_storage()
    return value.device, storage._cdata, storage.nbytes()


def retained_bytes(values):
    return sum(key[2] for key in {storage_key(value) for value in values})


class MetadataCache:
    """Own only pointer tables; no input, output or numerical buffers are shared."""

    def __init__(self, capacity=4):
        self.capacity = capacity
        self._tables = OrderedDict()
        self.hits = self.misses = 0

    def __reduce__(self):
        return type(self), (self.capacity,)

    def clear(self):
        self._tables.clear()

    def statistics(self):
        return {"hits": self.hits, "misses": self.misses, "tables": len(self._tables),
                "retained_bytes": retained_bytes(entry[0] for entry in self._tables.values())}

    def table(self, key, factory, device, owners=()):
        entry = self._tables.get(key)
        if entry is not None and all(reference() is owner for reference, owner in zip(entry[3], owners)):
            self.hits += 1
            self._tables.move_to_end(key)
            tensor, stream, event, _ = entry
            if device.type == "cuda":
                current = torch.cuda.current_stream(device)
                if stream != current:
                    current.wait_event(event)
                    tensor.record_stream(current)
            return tensor
        self.misses += 1
        tensor = torch.tensor(factory(), device=device, dtype=torch.int64)
        stream, event = None, None
        if device.type == "cuda":
            stream = torch.cuda.current_stream(device)
            event = torch.cuda.Event()
            event.record(stream)
        self._tables[key] = tensor, stream, event, tuple(ref(owner) for owner in owners)
        while len(self._tables) > self.capacity:
            self._tables.popitem(last=False)
        return tensor

    def state_table(self, states, device):
        key = ("states", tuple((id(state), state._storage_epoch, state.recipe_signature) for state in states), device)

        def rows():
            values = []
            for state in states:
                scale, history, updates, generations = (value.data_ptr() for value in
                                                        (state.scales, state.history, state.updates, state.generations))
                for role, name in enumerate(("activation", "weight", "grad")):
                    values.append((scale + role * 4, history + role * state.options.history_length * 4,
                                   updates + role * 8, generations + role * 8, state.options.history_length,
                                   state.options.margin, int(state.options.amax_algorithm == "most_recent"),
                                   int(torch.finfo(state.dtypes[name]).max)))
            return values

        return self.table(key, rows, device, states)
