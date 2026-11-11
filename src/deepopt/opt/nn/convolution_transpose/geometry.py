"""Transposed geometry with explicit bounds for the adjoint convolution."""

from dataclasses import dataclass

from ..convolution.geometry import ConvGeometry, _spatial_tuple


@dataclass(frozen=True)
class _AdjointGeometry(ConvGeometry):
    logical_output_spatial: tuple[int, ...] = ()

    @property
    def output_spatial(self):
        # Native output_padding may exceed stride when dilation permits it.
        # Explicit logical bounds mask the otherwise extra virtual positions.
        return self.logical_output_spatial


@dataclass(frozen=True)
class ConvTransposeGeometry(ConvGeometry):
    output_padding: tuple[int, ...] = ()

    @classmethod
    def from_tensors(cls, x, weight, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        rank = weight.ndim - 2
        if rank not in (1, 2, 3) or x.ndim != rank + 2:
            raise ValueError("transposed convolution requires batched input of rank 3, 4, or 5")
        if type(groups) is not int or groups < 1:
            raise ValueError("groups must be a positive integer")
        if x.shape[1] != weight.shape[0] or weight.shape[0] % groups:
            raise ValueError("input channels must match weight and be divisible by groups")
        if any(v <= 0 for v in (*x.shape[1:], *weight.shape)):
            raise ValueError("channels, spatial dimensions and kernel sizes must be positive")
        stride = _spatial_tuple(stride, rank, "stride", minimum=1)
        padding = _spatial_tuple(padding, rank, "padding", minimum=0)
        dilation = _spatial_tuple(dilation, rank, "dilation", minimum=1)
        extra = _spatial_tuple(output_padding, rank, "output_padding", minimum=0)
        if any(o >= s and o >= d for o, s, d in zip(extra, stride, dilation)):
            raise ValueError("output padding must be smaller than either stride or dilation")
        result = cls(tuple(x.shape), tuple(weight.shape), stride, padding, dilation, groups,
                     output_padding=extra)
        if any(v <= 0 for v in result.output_spatial):
            raise ValueError("transposed convolution output spatial dimensions must be positive")
        return result

    @property
    def out_channels(self):
        return self.weight_shape[1] * self.groups

    @property
    def in_channels_per_group(self):
        return self.in_channels // self.groups

    @property
    def output_spatial(self):
        return tuple((n - 1) * s - 2 * p + d * (k - 1) + o + 1
                     for n, s, p, d, k, o in zip(self.input_spatial, self.stride, self.padding,
                                                self.dilation, self.kernel_size, self.output_padding))

    @property
    def weight_matrix_shape(self):
        return self.in_channels, self.out_channels_per_group * self.kernel_volume

    @property
    def pointwise(self):
        return super().pointwise and not any(self.output_padding)

    @property
    def adjoint(self):
        return _AdjointGeometry(self.output_shape, self.weight_shape, self.stride, self.padding,
                                self.dilation, self.groups, logical_output_spatial=self.input_spatial)
