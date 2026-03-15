import librosa
import soundfile as sf
import os
import numpy as np

# paths are relative so anyone who clones the repo can run this without changing anything
RAW_DIR = "data/raw"
OUT_DIR = "data/processed"

# i tried 3 seconds first but it kept cutting off calls mid-vocalization
# 5 seconds captures a full call for every species in this dataset
SEG_SEC = 5

# resampling everything to 22050hz because the macaulay library files
# are recorded at different sample rates and the model needs consistent input
SR = 22050

# looked these up from au & hastings "principles of marine bioacoustics"
# and cross-referenced with the macaulay library species pages
# the beaked whale range is actually 20khz+ in reality but our sample rate
# only goes up to 11025hz (nyquist theorem: max freq = sample_rate / 2)
# so we use 2000-8000 as a proxy - this is a known weakness in this approach
SPECIES_FREQ = {
    "blue_whale":         (10, 200),    # infrasonic, very low rumbles
    "fin_whale":          (15, 150),    # 20hz pulses
    "humpback_whale":     (80, 4000),   # complex songs, widest range
    "sperm_whale":        (100, 8000),  # clicks and codas
    "orca":               (500, 8000),  # whistles + echolocation clicks
    "pilot_whale":        (1000, 8000), # high-freq whistles
    "false_killer_whale": (1000, 8000), # similar call structure to pilot whale
    "beaked_whale":       (2000, 8000), # proxy range - real calls are above our sample rate ceiling
}


def has_whale_energy(chunk, sr, species, threshold=0.02):
    """
    Returns True if this chunk probably contains whale vocalizations.

    The idea: whale calls concentrate energy in a specific frequency band.
    If we compute what fraction of total signal energy falls in that band,
    chunks that are mostly background noise (boat engines, researcher audio,
    snapping shrimp) will have a low ratio and get filtered out.

    threshold=0.02 means we require at least 2% of total energy in the whale band.
    I tested 0.01 and 0.05 on a small sample - 0.01 let through too much noise,
    0.05 was dropping real calls. 0.02 felt like the right tradeoff.

    Limitation: this won't separate a whale call that overlaps with background noise.
    It only removes chunks where there's essentially no whale signal at all.
    Full source separation would require something like a trained U-Net which is
    out of scope here.
    """
    freq_range = SPECIES_FREQ.get(species, (10, 8000))

    # stft converts the waveform into a 2d array of (frequency, time) energy values
    S = np.abs(librosa.stft(chunk))
    freqs = librosa.fft_frequencies(sr=sr)

    # find the bin indices that correspond to this species' frequency range
    low_bin = np.searchsorted(freqs, freq_range[0])
    high_bin = np.searchsorted(freqs, freq_range[1])

    band_energy = S[low_bin:high_bin, :].mean()
    total_energy = S.mean()

    # completely silent chunks happen at the end of some recordings
    if total_energy < 1e-6:
        return False

    return (band_energy / total_energy) > threshold


def slice_file(filepath, species):
    """
    Slices one mp3 into 5-second wav segments.
    Only keeps segments that pass the frequency energy filter.

    Naming convention: {species}_{source_filename}_{index:04d}.wav
    The source filename is kept in the segment name so we can always
    trace a segment back to the original recording if something looks wrong.
    """
    out_dir = os.path.join(OUT_DIR, species)
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading {os.path.basename(filepath)}...")
    y, _ = librosa.load(filepath, sr=SR, mono=True)

    seg_len = SR * SEG_SEC
    total_segs = len(y) // seg_len
    kept = 0
    skipped = 0

    for i in range(total_segs):
        chunk = y[i * seg_len : (i + 1) * seg_len]

        if not has_whale_energy(chunk, SR, species):
            skipped += 1
            continue

        out_name = f"{species}_{os.path.splitext(os.path.basename(filepath))[0]}_{i:04d}.wav"
        sf.write(os.path.join(out_dir, out_name), chunk, SR)
        kept += 1

    print(f"  -> kept {kept}, skipped {skipped} (no whale-band energy)")
    return kept


if __name__ == "__main__":
    counts = {}

    for fname in sorted(os.listdir(RAW_DIR)):
        if not fname.endswith(".mp3"):
            continue

        # filename format is {species}_{mlid}.mp3 e.g. blue_whale_ML120532.mp3
        # splitting on _ and dropping the last element gives us the species name
        species = "_".join(fname.split("_")[:-1])
        fpath = os.path.join(RAW_DIR, fname)

        n = slice_file(fpath, species)
        counts[species] = counts.get(species, 0) + n

    print("\n--- segment counts after filtering ---")
    for sp, n in sorted(counts.items()):
        print(f"  {sp}: {n}")
    print(f"  total: {sum(counts.values())}")
