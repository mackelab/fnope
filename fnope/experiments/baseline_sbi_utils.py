import torch


def pad_tensor(x, pad_width):
    """
    Pads the batched 1d or 2d tensor with the specified pad width on all sides
    with the neighboring values of the tensor.
    """
    if x.ndim==2:
        # Replicate the first and last values k times
        left_pad = x[:, 0:1].repeat(1, pad_width)  # shape (batch, pad_width)
        right_pad = x[:, -1:].repeat(1, pad_width)  # shape (batch, pad_width)

        # Concatenate the paddings with the original tensor
        x_padded = torch.cat([left_pad, x, right_pad], dim=1)
        
    else:
        raise ValueError("Invalid dimension of x. Expected batched 1D  tensor.")

    return x_padded


def perform_rfft_and_process(theta_raw, n_fft_modes, pad_width=0):
    """
    performs rfft, cuts to nfft_modes,
    and puts it to a 2*dim real tensor
    Args:
        theta_raw: the input tensor (batched 1D or 2D)
        n_fft_modes: the number of FFT modes to keep, int
        pad_width: the width of the padding to apply to the input tensor
                (only for 1D tensors)
    Returns:
        theta: the processed tensor (batched 1D or 2D)
        (H_pad, W_pad): the padding applied to the input tensor (only for 2D tensors)
    """
    # Assuring that the input is a batched 1 or 2d tensor
    if theta_raw.ndim != 2 and theta_raw.ndim != 3:
        raise ValueError("Input tensor must be batched 1 or 2d")
    
    if theta_raw.ndim == 2:
        theta_pad = pad_tensor(theta_raw, pad_width=pad_width)

        theta_fft = torch.fft.rfft(theta_pad, norm="forward")[:, :n_fft_modes]
        dim = min(n_fft_modes, theta_fft.shape[1])
        theta = torch.zeros(theta_raw.shape[0], dim * 2)
        theta[:, :dim] = theta_fft[:, :dim].real
        theta[:, dim:] = theta_fft[:, :dim].imag
        # to avoid sbi hassle:
        theta[:, dim] += torch.randn(theta.shape[0]) * 1e-10
        return theta
    if theta_raw.ndim == 3:
        fft_cropped, (H_pad, W_pad) = fft2_crop_lowpass_batched(theta_raw, (n_fft_modes, n_fft_modes))
        theta = make_real_2d(fft_cropped)
        return theta, (H_pad, W_pad)
       
    else:
        raise ValueError("Invalid dimension of x. Expected batched 1D or 2D tensor.")
    


def restore_fft(x_fft_real, size, pad_width, originial_dims = 1):
    """
    Restores the original 1d tensor from its FFT representation.
    The FFT representation is assumed to be in the form of a real-valued tensor
    with the first half representing the real part and the second half representing
    the imaginary part.
    Args:
        x_fft_real: the FFT representation of the original tensor (batched 1D)
        size: the original size of the tensor before padding (int or tuple)
        pad_width: the width of the padding applied to the original tensor (int or tuple)
                    for 2d it is the shape of the padded tensor
        originial_dims: the original dimensions of the tensor (1 or 2)
    """
    if originial_dims == 1:
        # assuring that first imaginary part is zero
        if isinstance(size, tuple):
            if len(size) != 1:
                raise ValueError("Invalid size. Expected a tuple of length 1 for 1D parameters.")
            size = size[0]
            
        x_fft_real[:,x_fft_real.shape[1] // 2] = 0

        x_complex = torch.complex(
            x_fft_real[:, : x_fft_real.shape[1] // 2],
            x_fft_real[:, x_fft_real.shape[1] // 2 :],
        )

        x_restore = torch.fft.irfft(x_complex, n=size + (2 * pad_width), norm="forward")
        if pad_width > 0:
            x_restore = x_restore[:, pad_width:-pad_width]

    elif originial_dims == 2:
        x_fft_complexed = recover_complex_from_real_2d(x_fft_real)
        x_restore = ifft2_from_cropped_fft(x_fft_complexed, size, pad_width)

    else:
        raise ValueError("Invalid dimension of x. Expected target to be batched 1D or 2D tensor.")

    return x_restore


### For 2D FFTs ###

def fft2_crop_lowpass_batched(images, num_components):
    """
    Compute the FFT2 of a batch of real 2D images with reflect padding,
    and return the cropped low-frequency components.

    Args:
        images (Tensor): shape (B, H, W) — real-valued images
        num_components (tuple): (H_crop, W_crop) — number of low-freq components to retain

    Returns:
        fft_cropped (Tensor): (B, H_crop, W_crop) — complex-valued cropped FFT
        orig_shape (tuple): (H, W) — original spatial size of the image
        pad_shape (tuple): (Hp, Wp) — padded shape used for FFT
    """
    B, H, W = images.shape
    H_keep, W_keep = num_components

    # Reflect padding
    pad = (W // 2, W // 2, H // 2, H // 2)
    padded = torch.nn.functional.pad(images, pad=pad, mode='replicate')  # (B, Hp, Wp)
    Hp, Wp = padded.shape[-2:]

    # FFT and shift
    fft = torch.fft.fft2(padded, norm="forward")                          # (B, Hp, Wp)
    # add small noise to avoid sbi issues
    fft[:, 0,0] += (torch.randn(images.shape[0]) * 1e-10 * 1j).to(fft.device)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))   # center frequencies

    # Crop center
    cy, cx = Hp // 2, Wp // 2
    sy, sx = H_keep // 2, W_keep // 2
    fft_cropped = fft_shifted[:, cy - sy:cy + sy, cx - sx:cx + sx]  # (B, H_crop, W_crop)

    return fft_cropped, (Hp, Wp)


def ifft2_from_cropped_fft(fft_cropped, orig_shape, pad_shape):
    """
    Reconstruct a real-valued image from cropped FFT2 components.

    Args:
        fft_cropped (Tensor): shape (B, H_crop, W_crop) — low-frequency FFT
        orig_shape (tuple): (H, W) — original unpadded spatial size
        pad_shape (tuple): (Hp, Wp) — shape used during FFT padding

    Returns:
        filtered (Tensor): real-valued image of shape (B, H, W)
    """
    B, H_crop, W_crop = fft_cropped.shape
    Hp, Wp = pad_shape
    H, W = orig_shape

    # Compute center and half-size
    cy, cx = Hp // 2, Wp // 2
    sy, sx = H_crop // 2, W_crop // 2

    # Zero-pad cropped FFT into full-size FFT-shifted array
    fft_padded = torch.zeros((B, Hp, Wp), dtype=torch.complex64, device=fft_cropped.device)
    fft_padded[:, cy - sy:cy + sy, cx - sx:cx + sx] = fft_cropped

    # Inverse shift and IFFT
    fft_unshifted = torch.fft.ifftshift(fft_padded, dim=(-2, -1))
    filtered_full = torch.fft.ifft2(fft_unshifted, norm="forward").real  # real-valued

    # Crop back to original size
    start_y = Hp // 2 - H // 2
    start_x = Wp // 2 - W // 2
    return filtered_full[:, start_y:start_y + H, start_x:start_x + W].real


def make_real_2d(x):
    """
    Make a complex tensor real by copying imaginary part into second half 
    and flattening the tensor.
    """
    dim2 = x.shape[-1]
    dim1 = x.shape[-2]
    x_real = torch.zeros(x.shape[0], dim1* dim2 * 2)
    x_real[:, :dim1*dim2] = x.real.flatten(start_dim=1)
    x_real[:, dim1*dim2:] = x.imag.flatten(start_dim=1)

    return x_real

def recover_complex_from_real_2d(x_real):
    """
    Recover a complex tensor from the real-valued representation created by `make_real`.
    assuming that it comes from a batched nxn tensor

    Args:
        x_real (Tensor): shape (B, 2 * H * W) — real+imag concatenated

    Returns:
        Tensor of shape (B, H, W), dtype=complex64
    """
    B, D = x_real.shape
    H = int((D/2)**0.5)
    W = H
    assert D == 2 * H * W, "Input shape does not match original dimensions"

    real_part = x_real[:, :H * W].reshape(B, H, W)
    imag_part = x_real[:, H * W:].reshape(B, H, W)
    return torch.complex(real_part, imag_part)

