# Libraries
import time
import torch
from pesq import pesq
import shutil
import random
import librosa
import torchaudio
import numpy as np
import pandas as pd
import torch.nn as nn
import soundfile as sf
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Functions
# Measure the duration code has been running, used to measure duration of each epoch
def timeit(start_time, end_time=None):
    if (end_time == None):
        end_time = time.time()
    time_difference = end_time - start_time
    minutes = int(time_difference // 60)  # Integer division to get minutes
    seconds = int(time_difference % 60)  # Modulo to get remaining seconds
    output = ""
    if (minutes):
        output = output + f"{minutes} minutes "
    output = output + f"{seconds} seconds"
    return output

# estimates the memory of model its optimizer and gradients, used to check total memory needed excluding data in GPU
def estimate_training_memory(model, optimizer, dtype=torch.float32):
    param_size = sum(p.numel() for p in model.parameters())
    param_memory = param_size * dtype.itemsize

    grad_memory = param_memory

    # Optimizer state memory (depends on optimizer)
    if isinstance(optimizer, torch.optim.Adam) or isinstance(optimizer, torch.optim.AdamW):
        optim_memory = param_memory * 4
    elif isinstance(optimizer, torch.optim.SGD):
        optim_memory = param_memory * 2
    else:
        optim_memory = param_memory * 2

    total_memory = param_memory + grad_memory + optim_memory
    return total_memory / (1024 ** 2) # to MB

# Computes ERLE Score, used for performance evaluation while training
def compute_erle(clean_wave, predicted_wave):
    min_length = min(clean_wave.shape[-1], predicted_wave.shape[-1])
    clean_wave = clean_wave[:min_length]
    predicted_wave = predicted_wave[:min_length]
    
    ephsilon = 1e-10
    clean_power = clean_wave ** 2
    predicted_power = predicted_wave ** 2
    avg_clean_power = np.mean(clean_power)
    avg_predicted_power = np.mean(predicted_power) + ephsilon
    erle = 10 * np.log10(avg_clean_power / avg_predicted_power)

    return erle


# Convert mag and angle to a wave file and write into a file, used in validation
def to_wave(mag, angle, filename="filename_NA", n_fft=512, hop_length=256, sr=16000, write=0):
    # real = mag * np.cos(angle)
    # imag = mag * np.sin(angle)
    # complex = real.astype(np.complex64) + 1j * imag.astype(np.complex64)
    complex = mag * angle # the value inside angle variable is the phase format, hence above calculation is not needed
    audio = librosa.istft(complex, n_fft=n_fft, hop_length=hop_length)
    if (write == 1):
        sf.write(f"./SampleFiles/{filename}.wav", audio, sr)
    return audio

# Plots the spectogram, plots spectogram of mixed, far1, far2, near and output from model
def plot_spectograms(data, titles, sr=16000):
    num_plots = len(data)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 5 * num_plots))
    for i, spect in enumerate(data):
        spect_db = librosa.amplitude_to_db(np.abs(spect), ref=np.max) # amplitude to dB
        img = librosa.display.specshow(spect_db, sr=sr, x_axis='time', y_axis='log', hop_length=256, ax=axes[i])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Frequency (Hz)")
        fig.colorbar(img, ax=axes[i], format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

# loads a signle row of files (similar to our custom dataset class), used to show the output of the model in detail to guide
def load_files(mixed_path, far_path1, far_path2, near_path, n_fft=512, hop_length=256, device='cuda', max_frames=512, write=1, window=0):
    mixed_wave, _ = torchaudio.load(mixed_path)
    far_wave1, _ = torchaudio.load(far_path1)
    far_wave2, _ = torchaudio.load(far_path2)
    near_wave, _ = torchaudio.load(near_path)

    # STFT w/o windowing
    if (window == 0):
        mixed_stft = torch.stft(mixed_wave.squeeze(0), n_fft=n_fft, hop_length=hop_length, return_complex=True, normalized=True, center=False)
        far_stft1 = torch.stft(far_wave1.squeeze(0), n_fft=n_fft, hop_length=hop_length, return_complex=True, normalized=True, center=False)
        far_stft2 = torch.stft(far_wave2.squeeze(0), n_fft=n_fft, hop_length=hop_length, return_complex=True, normalized=True, center=False)
        near_stft = torch.stft(near_wave.squeeze(0), n_fft=n_fft, hop_length=hop_length, return_complex=True, normalized=True, center=False)

    # STFT w windowing
    else:
        mixed_stft = torch.stft(mixed_wave.squeeze(0), n_fft=n_fft, hop_length=hop_length,
                            center=False, return_complex=True, normalized=True,
                            win_length=n_fft, window=torch.hann_window(window_length=n_fft))
        far_stft1 = torch.stft(far_wave1.squeeze(0), n_fft=n_fft, hop_length=hop_length,
                            center=False, return_complex=True, normalized=True,
                            win_length=n_fft, window=torch.hann_window(window_length=n_fft))
        far_stft2 = torch.stft(far_wave2.squeeze(0), n_fft=n_fft, hop_length=hop_length,
                            center=False, return_complex=True, normalized=True,
                            win_length=n_fft, window=torch.hann_window(window_length=n_fft))
        near_stft = torch.stft(near_wave.squeeze(0), n_fft=n_fft, hop_length=hop_length,
                            center=False, return_complex=True, normalized=True,
                            win_length=n_fft, window=torch.hann_window(window_length=n_fft))
    
    mixed_mag = torch.abs(mixed_stft)
    far_mag1 = torch.abs(far_stft1)
    far_mag2 = torch.abs(far_stft2)
    near_mag = torch.abs(near_stft)
    phase = torch.exp(1j*torch.angle(near_stft))

     
    # print(mixed_mag.size(-1)) # max is 1023
    if mixed_mag.size(-1) < max_frames:
        padding = max_frames - mixed_mag.size(-1)
        mixed_mag = torch.nn.functional.pad(mixed_mag, (0, padding))
        far_mag1 = torch.nn.functional.pad(far_mag1, (0, padding))
        far_mag2 = torch.nn.functional.pad(far_mag2, (0, padding))
        near_mag = torch.nn.functional.pad(near_mag, (0, padding))
        phase = torch.nn.functional.pad(phase, (0, padding))
    elif mixed_mag.size(-1) > max_frames:
        mixed_mag = mixed_mag[:, :max_frames]
        far_mag1 = far_mag1[:, :max_frames]
        far_mag2 = far_mag2[:, :max_frames]
        near_mag = near_mag[:, :max_frames]
        phase = phase[:, :max_frames]

    combined = torch.cat((far_mag1, far_mag2, mixed_mag), dim=0)
    
    # print(mixed_wave.shape, mixed_stft.shape, mixed_mag.shape, combined.shape)
    if (write == 1):
        shutil.copy2(mixed_path, "./SampleFiles/mixed.wav")
        shutil.copy2(near_path, "./SampleFiles/near.wav")
        shutil.copy2(far_path1, "./SampleFiles/far1.wav")
        shutil.copy2(far_path2, "./SampleFiles/far2.wav")
    return combined.to(device), near_mag, mixed_mag, far_mag1, far_mag2, phase, near_wave.squeeze(0).numpy()

# Get random samples
def random_samples(val_dataset, batch_size=64):
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    r = random.randint(2, 15)
    for i, (mixed_mag, far_mag_1, far_mag_2, near_mag, near_phase) in enumerate(val_loader):
        if (i < r):
            continue
        with torch.no_grad():
            # combined_input = torch.cat((far_mag_1.unsqueeze(1), far_mag_2.unsqueeze(1), mixed_mag.unsqueeze(1)), dim=1)
            mixed_mag = mixed_mag.cpu().numpy()
            far_mag_1 = far_mag_1.cpu().numpy()
            far_mag_2 = far_mag_2.cpu().numpy()
            near_mag = near_mag.cpu().numpy()
            near_phase = near_phase.cpu().numpy()
            return (mixed_mag, far_mag_1, far_mag_2, near_mag, near_phase)

# return ERLE and PESQ
def erle_pesq(outputs, near_wave, near_phase):
    pesq_avg, erle_avg = 0, 0
    l = len(near_wave)
    for i in range(l):
        near_wave_numpy = near_wave[i].squeeze(0).numpy()
        output_wave = to_wave(outputs[i], near_phase[i])
        erle_score = compute_erle(near_wave_numpy, output_wave)
        pesq_score = pesq(16000, near_wave_numpy, output_wave)
        pesq_avg += pesq_score
        erle_avg += erle_score
    return erle_avg/l, pesq_avg/l

def sisnr_loss(estimated, target, eps=1e-8):
    """
    Scale-Invariant Signal-to-Noise Ratio (SISNR) loss.
    """
    target = target.flatten(start_dim=1)
    estimated = estimated.flatten(start_dim=1)
    # print(target.shape, estimated.shape)
    s_target = target
    s_estimated = estimated

    s_target_norm = torch.sum(s_target**2, dim=-1, keepdim=True)
    proj_target = torch.sum(s_estimated * s_target, dim=-1, keepdim=True) * s_target / (s_target_norm + eps)
    noise = s_estimated - proj_target
    # print(s_target_norm.shape, proj_target.shape, noise.shape)

    loss = 10 * torch.log10(torch.sum(proj_target**2, dim=-1) / (torch.sum(noise**2, dim=-1) + eps))
    # print(loss.shape)
    return -torch.mean(loss)


# Read and return CSV file
def read_csv_data(filepath):
    df = pd.read_csv(filepath)
    return df