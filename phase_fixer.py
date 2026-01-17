import argparse
import torch
import torchaudio
import os

def frequency_blend_phases(phase1, phase2, freq_bins, low_cutoff=500, high_cutoff=5000, base_factor=0.25, scale_factor=1.85):
    """
    Blend two phase arrays with different weights depending on frequency.
    
    Parameters:
        phase1: Tensor of shape (frequency_bins, time_frames) - First phase matrix.
        phase2: Tensor of shape (frequency_bins, time_frames) - Second phase matrix.
        freq_bins: Tensor of shape (frequency_bins,) - Frequencies corresponding to bins.
        low_cutoff: int - Frequency below which blend_factor is base_factor.
        high_cutoff: int - Frequency above which blend_factor is base_factor + scale_factor.
        base_factor: float - The starting blend factor for low frequencies.
        scale_factor: float - The difference in blend factor between low and high frequencies.
    Returns:
        blended_phase: Tensor of shape (frequency_bins, time_frames).
    """
    # Validate input dimensions
    if phase1.shape != phase2.shape:
        raise ValueError("phase1 and phase2 must have the same shape.")
    if len(freq_bins) != phase1.shape[0]:
        raise ValueError("freq_bins must have the same length as the number of frequency bins in phase1 and phase2.")
    if low_cutoff >= high_cutoff:
        raise ValueError("low_cutoff must be less than high_cutoff.")

    # Initialize blended phase
    blended_phase = torch.zeros_like(phase1)

    # Compute blend factors for all frequencies
    blend_factors = torch.zeros_like(freq_bins)

    # Below low_cutoff: blend factor is base_factor
    blend_factors[freq_bins < low_cutoff] = base_factor

    # Above high_cutoff: blend factor is base_factor + scale_factor
    blend_factors[freq_bins > high_cutoff] = base_factor + scale_factor

    # Between low_cutoff and high_cutoff: interpolate linearly
    in_range_mask = (freq_bins >= low_cutoff) & (freq_bins <= high_cutoff)
    blend_factors[in_range_mask] = base_factor + scale_factor * (
        (freq_bins[in_range_mask] - low_cutoff) / (high_cutoff - low_cutoff)
    )

    # Apply blend factors to each frequency bin
    for i in range(phase1.shape[0]):
        blended_phase[i, :] = (1 - blend_factors[i]) * phase1[i, :] + blend_factors[i] * phase2[i, :]

    # Wrap phase to the range [-π, π]
    blended_phase = torch.remainder(blended_phase + torch.pi, 2 * torch.pi) - torch.pi

    return blended_phase

def transfer_magnitude_phase(source_file, target_file, transfer_magnitude=True, transfer_phase=True, low_cutoff=500, high_cutoff=5000, scale_factor=1.85):
    # Determine output path with "(Corrected)" suffix
    target_dir = os.path.dirname(target_file)
    target_name, target_ext = os.path.splitext(os.path.basename(target_file))
    output_file = os.path.join(target_dir, f"{target_name}_phasefixer{target_ext}")

    # Load source and target audio in stereo
    source_audio, source_sr = librosa.load(source_file, sr=None, mono=False)
    target_audio, target_sr = librosa.load(target_file, sr=None, mono=False)

    # Ensure sample rates match
    if source_sr != target_sr:
        raise ValueError("Sample rates of source and target audio files must match.")
    
    # STFT settings
    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft)

    # Compute STFTs for each channel
    source_stfts = torch.stft(source_waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, pad_mode="reflect")
    target_stfts = torch.stft(target_waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True, pad_mode="reflect")

    # Frequency bins
    freqs = torch.linspace(0, source_sr // 2, steps=n_fft // 2 + 1)

    # Process each channel independently
    modified_stfts = []
    for source_stft, target_stft in zip(source_stfts, target_stfts):
        source_mag, source_phs = torch.abs(source_stft), torch.angle(source_stft)
        target_mag, target_phs = torch.abs(target_stft), torch.angle(target_stft)

        # Transfer magnitude
        modified_stft = target_stft.clone()
        if transfer_magnitude:
            modified_stft = source_mag * torch.exp(1j * torch.angle(modified_stft))

        # Transfer or blend phase
        if transfer_phase:
            blended_phase = frequency_blend_phases(target_phs, source_phs, freqs, low_cutoff, high_cutoff, scale_factor)
            modified_stft = torch.abs(modified_stft) * torch.exp(1j * blended_phase)

        modified_stfts.append(modified_stft)

    # Convert modified STFTs back to time domain
    modified_waveform = torch.istft(
        torch.stack(modified_stfts),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=source_waveform.size(1)
    )

    # Save the modified audio to a file
    dtype = torch.float32
    torchaudio.save(output_file, modified_waveform, target_sr, encoding="PCM_S", bits_per_sample=32)
    print(f"Modified audio saved as {output_file}")

if __name__ == "__main__":
    # Usage
    parser = argparse.ArgumentParser(description="Transfer magnitude and/or phase between audio files.")
    parser.add_argument("--source_file", required=True, help="Path to the base folder containing instrumental files (kim).")
    parser.add_argument("--target_file", required=True, help="Path to the folder containing corresponding unwa files.")
    parser.add_argument("--transfer_magnitude", action='store_true', help="")
    parser.add_argument("--transfer_phase", action='store_true', help="")
    parser.add_argument("--low_cutoff", type=int, default=500, help="Low cutoff frequency for phase blending.")
    parser.add_argument("--high_cutoff", type=int, default=5000, help="High cutoff frequency for phase blending.")
    parser.add_argument("--scale_factor", type=float, default=1.85, help="Scale factor for phase blending.")

    # Adjust low and high cutoff frequencies if needed
    transfer_magnitude_phase(args.source_file, args.target_file, transfer_magnitude=args.transfer_magnitude, transfer_phase=args.transfer_phase, low_cutoff=args.low_cutoff, high_cutoff=args.high_cutoff, scale_factor=args.scale_factor)
