import os
import random
import torch
import torchaudio
import torch.utils.data as data

class LibriSpeechDataset(data.Dataset):
    def __init__(
        self,
        root_dir="/data/LibriSpeech",
        subset="train-clean-100",
        sample_rate=24000,
        segment_seconds=1.0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.segment_length = int(sample_rate * segment_seconds)

        self.files = []
        subset_dir = os.path.join(root_dir, subset)

        for root, _, filenames in os.walk(subset_dir):
            for fname in filenames:
                if fname.endswith(".flac"):
                    self.files.append(os.path.join(root, fname))

        assert len(self.files) > 0, "No audio files found."

        self.resampler = None

    def __len__(self):
        return len(self.files)

    def _load_audio(self, path):
        wav, sr = torchaudio.load(path)  # (C, T)

        # Convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=self.sample_rate
                )
            wav = self.resampler(wav)

        return wav

    def _random_crop(self, wav):
        T = wav.size(1)

        if T < self.segment_length:
            pad = self.segment_length - T
            wav = torch.nn.functional.pad(wav, (0, pad))
            return wav

        start = random.randint(0, T - self.segment_length)
        return wav[:, start : start + self.segment_length]

    def __getitem__(self, idx):
        path = self.files[idx]
        wav = self._load_audio(path)
        wav = self._random_crop(wav)
        return wav

if __name__ == '__main__':


    dataset = LibriSpeechDataset(
        root_dir="data/LibriSpeech",
        subset="train-clean-100",
        sample_rate=24000,
        segment_seconds=1.0,
    )

    print(dataset)
