import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # no display needed, just saving files
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil

PROCESSED_DIR = "data/processed"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

# 80/10/10 split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# rest goes to test

SR = 22050
IMG_SIZE = (2.24, 2.24)  # 224x224 at 100dpi = resnet18 input size

random.seed(42)  # reproducible splits


def wav_to_spectrogram(wav_path, out_path):
    y, _ = librosa.load(wav_path, sr=SR)

    # mel spectrogram - these settings work well for cetacean audio
    # n_mels=128 gives good freq resolution, fmax=8000 covers most whale calls
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=128,
                                        hop_length=512, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=IMG_SIZE, dpi=100)
    ax.axis('off')
    librosa.display.specshow(S_db, ax=ax, cmap='magma',
                              sr=SR, hop_length=512, fmax=8000)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def process_species(species):
    species_dir = os.path.join(PROCESSED_DIR, species)
    wavs = [f for f in os.listdir(species_dir) if f.endswith('.wav')]

    if len(wavs) == 0:
        print(f"  no wavs found for {species}, skipping")
        return

    random.shuffle(wavs)

    n = len(wavs)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    splits = {
        'train': wavs[:n_train],
        'val':   wavs[n_train:n_train + n_val],
        'test':  wavs[n_train + n_val:]
    }

    dirs = {'train': TRAIN_DIR, 'val': VAL_DIR, 'test': TEST_DIR}

    counts = {}
    for split, files in splits.items():
        out_dir = os.path.join(dirs[split], species)
        os.makedirs(out_dir, exist_ok=True)
        for fname in files:
            wav_path = os.path.join(species_dir, fname)
            png_name = fname.replace('.wav', '.png')
            out_path = os.path.join(out_dir, png_name)
            try:
                wav_to_spectrogram(wav_path, out_path)
            except Exception as e:
                print(f"  failed on {fname}: {e}")
        counts[split] = len(files)

    print(f"  {species}: {counts['train']} train, {counts['val']} val, {counts['test']} test")


if __name__ == "__main__":
    species_list = os.listdir(PROCESSED_DIR)
    species_list = [s for s in species_list if os.path.isdir(os.path.join(PROCESSED_DIR, s))]

    print(f"found {len(species_list)} species: {species_list}\n")

    for species in sorted(species_list):
        print(f"processing {species}...")
        process_species(species)

    print("\ndone")
