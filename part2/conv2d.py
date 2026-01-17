import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size

    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Tile dimensions based on hardware constraints
    TILE_C_IN = nl.tile_size.pmax  # 128
    TILE_C_OUT = nl.tile_size.pmax  # 128
    TILE_WIDTH = nl.tile_size.gemm_moving_fmax  # 512
    TILE_HEIGHT = out_height  # Process full height at once

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # Reshape input: [in_channels, input_height, input_width] -> [in_channels, input_height * input_width]
        x_reshaped = X[b].reshape((in_channels, input_height * input_width))

        # Loop over output channel tiles
        for oc_tile in nl.affine_range(out_channels // TILE_C_OUT):
            oc_start = oc_tile * TILE_C_OUT
            oc_end = (oc_tile + 1) * TILE_C_OUT

            # Loop over width tiles
            for w_tile in nl.affine_range(out_width // TILE_WIDTH):
                w_start = w_tile * TILE_WIDTH
                w_end = (w_tile + 1) * TILE_WIDTH

                # Initialize PSUM accumulator for convolution results
                # Shape: [TILE_C_OUT, TILE_HEIGHT, TILE_WIDTH]
                conv_psum = nl.zeros((TILE_C_OUT, TILE_HEIGHT, TILE_WIDTH), dtype=X.dtype, buffer=nl.psum)

                # Loop over filter positions (the algorithmic core: conv via shifted matmuls)
                for fh in nl.affine_range(filter_height):
                    for fw in nl.affine_range(filter_width):
                        # Load weights to SBUF and transpose
                        # W slice: [oc_start:oc_end, :, fh, fw] has shape [TILE_C_OUT, in_channels]
                        # Need to load as [in_channels, TILE_C_OUT] for nc_matmul
                        w_sbuf = nl.ndarray((TILE_C_OUT, in_channels), dtype=W.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(src=W[oc_start:oc_end, :, fh, fw], dst=w_sbuf)

                        # Load input to SBUF - extract shifted columns
                        x_sbuf = nl.ndarray((in_channels, TILE_HEIGHT * TILE_WIDTH), dtype=X.dtype, buffer=nl.sbuf)
                        for h in nl.affine_range(TILE_HEIGHT):
                            for w in nl.affine_range(TILE_WIDTH):
                                # Compute the column index in the flattened input
                                col_idx = (h + fh) * input_width + (w + fw)
                                nisa.dma_copy(src=x_reshaped[:, col_idx:col_idx+1], dst=x_sbuf[:, h*TILE_WIDTH + w:h*TILE_WIDTH + w + 1])

                        # Perform matrix multiplication using Tensor Engine
                        # w_sbuf: [TILE_C_OUT, in_channels]
                        # x_sbuf: [in_channels, TILE_HEIGHT * TILE_WIDTH]
                        # Result: [TILE_C_OUT, TILE_HEIGHT * TILE_WIDTH]
                        matmul_result = nisa.nc_matmul(w_sbuf, x_sbuf)

                        # Reshape to [TILE_C_OUT, TILE_HEIGHT, TILE_WIDTH] and accumulate
                        matmul_reshaped = matmul_result.reshape((TILE_C_OUT, TILE_HEIGHT, TILE_WIDTH))
                        conv_psum = nisa.tensor_tensor(conv_psum, matmul_reshaped, op=nl.add)

                # Add bias after convolution is complete
                bias_slice = bias[oc_start:oc_end]
                bias_sbuf = nl.ndarray((TILE_C_OUT, TILE_HEIGHT, TILE_WIDTH), dtype=X.dtype, buffer=nl.sbuf)
                # Broadcast bias across spatial dimensions
                for h in nl.affine_range(TILE_HEIGHT):
                    for w in nl.affine_range(TILE_WIDTH):
                        bias_sbuf[:, h, w] = bias_slice

                conv_psum = nisa.tensor_tensor(conv_psum, bias_sbuf, op=nl.add)

                # Copy result from PSUM to SBUF
                conv_sbuf = nisa.tensor_copy(conv_psum)

                # Apply maxpool if needed and store to HBM
                if pool_size == 2:
                    pooled_height = TILE_HEIGHT // pool_size
                    pooled_width = TILE_WIDTH // pool_size
                    pooled_sbuf = nl.ndarray((TILE_C_OUT, pooled_height, pooled_width), dtype=X.dtype, buffer=nl.sbuf)

                    for ph in nl.affine_range(pooled_height):
                        for pw in nl.affine_range(pooled_width):
                            # Extract the 2x2 pool region
                            pool_region = conv_sbuf[:, ph*pool_size:(ph+1)*pool_size, pw*pool_size:(pw+1)*pool_size]
                            # Compute max over the pool region for each channel
                            # Reshape to [TILE_C_OUT, pool_size*pool_size] then reduce
                            pool_flat = pool_region.reshape((TILE_C_OUT, pool_size * pool_size))
                            max_val = nisa.tensor_reduce(nl.max, pool_flat, axis=1)
                            pooled_sbuf[:, ph, pw] = max_val

                    # Store pooled result to HBM
                    nisa.dma_copy(src=pooled_sbuf,
                                 dst=X_out[b, oc_start:oc_end, :, w_start//pool_size:w_end//pool_size])
                else:
                    # Store result without pooling
                    nisa.dma_copy(src=conv_sbuf,
                                 dst=X_out[b, oc_start:oc_end, :, w_start:w_end])

    return X_out

