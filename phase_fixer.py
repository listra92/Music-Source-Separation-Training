import os
import sys
import argparse
import librosa
import numpy as np
import soundfile as sf

def frequency_blend_phases(phase1, phase2, freq_bins, low_cutoff=500, high_cutoff=5000):
    """Blend two phase arrays with different weights depending on frequency."""
    blended_phase = np.zeros_like(phase1)
    for i, freq in enumerate(freq_bins):
        if freq < low_cutoff:
            blend_factor = 0.1  # Mostly keep original phase for low frequencies
        elif freq > high_cutoff:
            blend_factor = 0.9  # Strongly blend with new phase for high frequencies
        else:
            blend_factor = 0.1 + 0.8 * ((freq - low_cutoff) / (high_cutoff - low_cutoff))
        blended_phase[i, :] = (1 - blend_factor) * phase1[i, :] + blend_factor * phase2[i, :]
    return blended_phase

def transfer_magnitude_phase(source_file, target_file, transfer_magnitude=True, transfer_phase=True, low_cutoff=500, high_cutoff=5000, subtype='FLOAT'):
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
    
    # Handle stereo channels separately
    source_stft_left = librosa.stft(source_audio[0, :])
    source_stft_right = librosa.stft(source_audio[1, :])
    target_stft_left = librosa.stft(target_audio[0, :])
    target_stft_right = librosa.stft(target_audio[1, :])

    # Extract magnitudes and phases
    source_magnitude_left, source_phase_left = np.abs(source_stft_left), np.angle(source_stft_left)
    source_magnitude_right, source_phase_right = np.abs(source_stft_right), np.angle(source_stft_right)
    target_magnitude_left, target_phase_left = np.abs(target_stft_left), np.angle(target_stft_left)
    target_magnitude_right, target_phase_right = np.abs(target_stft_right), np.angle(target_stft_right)

    # Frequency bins
    freqs = librosa.fft_frequencies(sr=source_sr, n_fft=source_stft_left.shape[0] * 2 - 1)

    # Initialize modified STFT arrays
    modified_stft_left = target_stft_left.copy()
    modified_stft_right = target_stft_right.copy()

    # Transfer magnitude
    if transfer_magnitude:
        modified_stft_left = source_magnitude_left * np.exp(1j * np.angle(modified_stft_left))
        modified_stft_right = source_magnitude_right * np.exp(1j * np.angle(modified_stft_right))

    # Transfer or blend phase with frequency-dependent weighting
    if transfer_phase:
        blended_phase_left = frequency_blend_phases(target_phase_left, source_phase_left, freqs, low_cutoff, high_cutoff)
        blended_phase_right = frequency_blend_phases(target_phase_right, source_phase_right, freqs, low_cutoff, high_cutoff)
        
        modified_stft_left = np.abs(modified_stft_left) * np.exp(1j * blended_phase_left)
        modified_stft_right = np.abs(modified_stft_right) * np.exp(1j * blended_phase_right)

    # Convert modified STFTs back to time domain for each channel
    modified_audio_left = librosa.istft(modified_stft_left, length=source_audio.shape[1])
    modified_audio_right = librosa.istft(modified_stft_right, length=source_audio.shape[1])
    
    # Combine left and right channels
    modified_audio = np.vstack((modified_audio_left, modified_audio_right))

    # Save the modified audio to a file
    sf.write(output_file, modified_audio.T, target_sr, subtype)
    print(f"Modified audio saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, help="source_file")
    parser.add_argument("--target_file", type=str, help="target_file")
    parser.add_argument("--subtype", default="FLOAT", type=str, help="subtype")
    parser.add_argument("--transfer_magnitude", default=False, type=bool, help="transfer_magnitude")
    parser.add_argument("--transfer_phase", default=True, type=bool, help="transfer_phase")
    parser.add_argument("--low_cutoff", default=500, type=int, help="low_cutoff")
    parser.add_argument("--high_cutoff", default=5000, type=int, help="high_cutoff")
    args = parser.parse_args()

    # Adjust low and high cutoff frequencies if needed
    transfer_magnitude_phase(args.source_file, args.target_file, args.transfer_magnitude, args.transfer_phase, args.low_cutoff, args.high_cutoff, args.subtype)
