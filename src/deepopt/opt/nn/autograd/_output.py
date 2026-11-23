"""Owned logical outputs for custom autograd functions with fresh workspaces."""


def restore_fresh_output(matrix, restore):
    """Restore a fresh executor allocation without returning an internal view.

    Only call for newly computed outputs, never inputs or saved state. Kernels
    have finished submitting their matrix descriptors before this metadata-only
    change. Returning an owned tensor permits ordinary downstream in-place ops
    without replacing this Function's custom backward or copying its output.
    """
    value = restore(matrix)
    if value._base is None:
        return value
    owner = matrix
    while owner._base is not None:
        owner = owner._base
    if owner.requires_grad:
        # A reference executor may return an existing differentiable tensor.
        # Preserve its ownership rather than changing external metadata.
        return value.clone()
    if (tuple(owner.shape) != tuple(value.shape) or tuple(owner.stride()) != tuple(value.stride())
            or owner.storage_offset() != value.storage_offset()):
        owner.as_strided_(value.shape, value.stride(), value.storage_offset())
    return owner
