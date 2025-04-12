import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader







class WaveDataset(Dataset):
    def __init__(self, mixed, far1, far2, near, n_fft=512, hop_length=256, max_frames=512, window=0, device='cuda'):
        self.mixed = mixed
        self.far1 = far1
        self.far2 = far2
        self.near = near
        self.files = mixed
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_frames = max_frames
        self.window = window
        self.device = device

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # read file -> stft -> abs of mag, remove phase -> return 4 values
        mixed_path = self.files[idx]
        far_path1 = self.far1[idx]
        far_path2 = self.far2[idx]
        near_path = self.near[idx]
        
        mixed_wave, _ = torchaudio.load(mixed_path)
        far_wave1, _ = torchaudio.load(far_path1)
        far_wave2, _ = torchaudio.load(far_path2)
        near_wave, _ = torchaudio.load(near_path)
    
        # STFT w/o windowing
        if (self.window == 0):
            mixed_stft = torch.stft(mixed_wave.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, normalized=True, center=False)
            far_stft1 = torch.stft(far_wave1.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, normalized=True, center=False)
            far_stft2 = torch.stft(far_wave2.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, normalized=True, center=False)
            near_stft = torch.stft(near_wave.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True, normalized=True, center=False)
    
        # STFT w windowing
        else:
            mixed_stft = torch.stft(mixed_wave.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length,
                                center=False, return_complex=True, normalized=True,
                                win_length=self.n_fft, window=torch.hann_window(window_length=self.n_fft))
            far_stft1 = torch.stft(far_wave1.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length,
                                center=False, return_complex=True, normalized=True,
                                win_length=self.n_fft, window=torch.hann_window(window_length=self.n_fft))
            far_stft2 = torch.stft(far_wave2.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length,
                                center=False, return_complex=True, normalized=True,
                                win_length=self.n_fft, window=torch.hann_window(window_length=self.n_fft))
            near_stft = torch.stft(near_wave.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length,
                                center=False, return_complex=True, normalized=True,
                                win_length=self.n_fft, window=torch.hann_window(window_length=self.n_fft))

        mixed_mag = torch.abs(mixed_stft)
        far_mag1 = torch.abs(far_stft1)
        far_mag2 = torch.abs(far_stft2)
        near_mag = torch.abs(near_stft)
        phase = torch.exp(1j*torch.angle(near_stft))
 
        # print(mixed_mag.size(-1)) # max is 1023
        if mixed_mag.size(-1) < self.max_frames:
            padding = self.max_frames - mixed_mag.size(-1)
            mixed_mag = torch.nn.functional.pad(mixed_mag, (0, padding))
            far_mag1 = torch.nn.functional.pad(far_mag1, (0, padding))
            far_mag2 = torch.nn.functional.pad(far_mag2, (0, padding))
            near_mag = torch.nn.functional.pad(near_mag, (0, padding))
            phase = torch.nn.functional.pad(phase, (0, padding))
        elif mixed_mag.size(-1) > self.max_frames:
            mixed_mag = mixed_mag[:, :self.max_frames]
            far_mag1 = far_mag1[:, :self.max_frames]
            far_mag2 = far_mag2[:, :self.max_frames]
            near_mag = near_mag[:, :self.max_frames]
            phase = phase[:, :self.max_frames]

        return mixed_mag.to(self.device), far_mag1.to(self.device), far_mag2.to(self.device), near_mag.to(self.device), phase, near_wave
