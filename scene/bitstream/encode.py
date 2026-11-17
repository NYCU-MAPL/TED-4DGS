# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: Theo Ladune <theo.ladune@orange.com>
#          Pierrick Philippe <pierrick.philippe@orange.com>


import math
import os
import subprocess
import torch
import time
from typing import Dict
from torch import Tensor

from scene.bitstream.header import DescriptorCoolChic, DescriptorNN, write_header
from scene.bitstream.range_coder import RangeCoder
from scene.arm import ArmMLP


def get_ac_max_val_latent(latent) -> int:
    """Return the maximum amplitude of the quantized latent variables.
    This allows to get the AC_MAX_VAL constant, used by the entropy coder. All
    symbols sent by the entropy coder will be in [-AC_MAX_VAL, AC_MAX_VAL - 1].

    Args:
        mode (CoolChicEncoder): Model storing the latent.

    Returns:
        int: The AC_MAX_VAL parameter.
    """
    # Setting visu to true allows to recover 2D mu, scale and latents
    latent = torch.cat([y_i.flatten() for y_i in latent], dim=0)

    # Compute AC_MAC_VAL
    AC_MAX_VAL = int(torch.ceil(latent.abs().max() + 2).item())
    return AC_MAX_VAL


@torch.no_grad()
def encode(
    model,
    bitstream_path: str,
    q_step_nn: DescriptorCoolChic,
) -> int:
    """Encode an image learned by model into a bitstream.

    Args:
        model (CoolChicEncoder): The trained overfitted encoder for the image.
        bitstream_path (str): Where to write the bitstream.
        q_step_nn (DescriptorCoolChic): Quantization step of the different NNs.
            See in header.py for the details.

    Returns:
        int: Rate in **byte**
    """

    start_time = time.time()
    # Ensure encoding/decoding replicability on different hardware
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Run on CPU for more reliability (particularly the ARM module)
    model = model.eval().to('cpu')

    if os.path.exists(bitstream_path):
        print(f'Found an already existing bitstream at {bitstream_path}... deleting!')
        subprocess.call(f'rm {bitstream_path}', shell=True)

    # ================= Encode the MLP into a bitstream file ================ #
    ac_max_val_nn = get_ac_max_val_nn(model, q_step_nn)
    range_coder_nn = RangeCoder(
        0,     # 0 because we don't have a ctx_row_col
        ac_max_val_nn
    )


    model.non_zero_pixel_ctx_index = model.non_zero_pixel_ctx_index.to('cpu')

    ac_max_val_latent = get_ac_max_val_latent(model)
    range_coder_latent = RangeCoder(model.n_ctx_rowcol, ac_max_val_latent)

    # Setting visu to true allows to recover 2D mu, scale and latents
    # Use AC_MAX_VAL to clamp the latent so they fit inside the expected range.
    # ! It should not be needed the AC_MAX_VAL parameter is computed to be the maximum
    # ! dynamic of the latent
    encoder_output = model.forward(get_proba_param = True, AC_MAX_VAL = range_coder_latent.AC_MAX_VAL)

    # Encode the different latent grids one after the other
    n_bytes_per_latent = []
    for i in range(model.latent_n_grids):
        current_mu = encoder_output.get('mu')[i]
        current_scale = encoder_output.get('scale')[i]
        current_scale = torch.round(current_scale)
        current_y = encoder_output.get('latent')[i]

        cur_latent_bitstream = f'{bitstream_path}_{i}'
        range_coder_latent.encode(
            cur_latent_bitstream,
            current_y.flatten().cpu(),
            current_mu.flatten().cpu(),
            current_scale.flatten().cpu(),
            (1, current_y.size()[-2], current_y.size()[-1]),
        )
        n_bytes_per_latent.append(os.path.getsize(cur_latent_bitstream))

    # Write the header
    header_path = f'{bitstream_path}_header'
    write_header(
        model,
        header_path,
        n_bytes_per_latent,
        q_step_index_nn,
        scale_index_nn,
        n_bytes_nn,
        ac_max_val_nn,
        ac_max_val_latent
    )

    # Concatenate everything inside a single file
    subprocess.call(f'rm -f {bitstream_path}', shell=True)
    subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
    subprocess.call(f'rm -f {header_path}', shell=True)

    for cur_module_name in ['arm', 'synthesis']:
        for parameter_type in ['weight', 'bias']:
            cur_bitstream = f'{bitstream_path}_{cur_module_name}_{parameter_type}'
            subprocess.call(f'cat {cur_bitstream} >> {bitstream_path}', shell=True)
            subprocess.call(f'rm -f {cur_bitstream}', shell=True)

    for i in range(model.latent_n_grids):
        cur_latent_bitstream = f'{bitstream_path}_{i}'
        subprocess.call(f'cat {cur_latent_bitstream} >> {bitstream_path}', shell=True)
        subprocess.call(f'rm -f {cur_latent_bitstream}', shell=True)

    real_rate_byte = os.path.getsize(bitstream_path)
    real_rate_bpp = real_rate_byte * 8 / (model.img_size[-1] * model.img_size[-2])
    print(f'Real rate        [kBytes]: {real_rate_byte / 1000:9.3f}')
    print(f'Real rate           [bpp]: {real_rate_bpp :9.3f}')

    elapsed = time.time() - start_time
    print(f'Encoding time: {elapsed:4.3f} sec')
    return real_rate_byte


