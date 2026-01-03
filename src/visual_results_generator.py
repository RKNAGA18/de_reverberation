import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from pesq import pesq

# ==========================================
# 🔴 STEP 1: FIX THESE PATHS
# Use the exact same paths you used for the metrics script!
# ==========================================
CLEAN_DIR = r"C:\Users\arjun\Desktop\hack\SGMSE_Eval\clean" # Example - Change this!
NOISY_DIR = r"C:\Users\arjun\Desktop\hack\SGMSE_Eval\noisy" # Example - Change this!
ENHANCED_DIR = r"C:\Users\arjun\Desktop\hack\n100_snr0.5\enhanced_output_n100_snr0.5"          # Or /enhanced_output
# ==========================================

# Helper function for alignment
def align_audio(ref, est):
    correlation = signal.correlate(ref, est, mode='full')
    lags = signal.correlation_lags(ref.size, est.size, mode='full')
    lag = lags[np.argmax(correlation)]
    if lag > 0:
        return np.pad(est, (lag, 0))[:ref.size]
    elif lag < 0:
        est = est[-lag:]
        return np.pad(est, (0, ref.size - est.size))
    return est

# Helper for SI-SDR
def calculate_sisdr(ref, est):
    eps = np.finfo(est.dtype).eps
    reference = ref.reshape(-1, 1)
    estimate = est.reshape(-1, 1)
    R = np.dot(reference.T, reference)
    if R == 0: return -100
    rho = np.dot(reference.T, estimate) / (R + eps)
    e_target = rho * reference
    e_res = estimate - e_target
    return 10 * np.log10((np.sum(e_target**2) + eps) / (np.sum(e_res**2) + eps))

# 1. Validation Logic
if not os.path.exists(CLEAN_DIR):
    print(f"❌ Error: The CLEAN_DIR path does not exist: {CLEAN_DIR}")
    print("👉 Please check the 'Copy File Path' in the Kaggle sidebar.")
elif not os.path.exists(ENHANCED_DIR):
    print(f"❌ Error: The ENHANCED_DIR path does not exist: {ENHANCED_DIR}")
else:
    # Find common files
    enhanced_files = set(os.listdir(ENHANCED_DIR))
    clean_files = set(os.listdir(CLEAN_DIR))
    valid_files = list(enhanced_files.intersection(clean_files))

    if len(valid_files) == 0:
        print("❌ Error: No matching filenames found!")
        print(f"Enhanced folder has {len(enhanced_files)} files (e.g., {list(enhanced_files)[:3]})")
        print(f"Clean folder has {len(clean_files)} files (e.g., {list(clean_files)[:3]})")
        print("👉 Check if your dataset has subfolders (e.g., /clean/tt/ instead of /clean/)")
    else:
        # 2. Success! Generate Plot
        TARGET_FILE = np.random.choice(valid_files)
        print(f"✅ Found {len(valid_files)} matches. Visualizing: {TARGET_FILE}")
        
        # Load and Process
        y_clean, sr = librosa.load(os.path.join(CLEAN_DIR, TARGET_FILE), sr=16000)
        y_noisy, _ = librosa.load(os.path.join(NOISY_DIR, TARGET_FILE), sr=16000)
        y_enhanced, _ = librosa.load(os.path.join(ENHANCED_DIR, TARGET_FILE), sr=16000)

        min_len = min(len(y_clean), len(y_noisy), len(y_enhanced))
        y_clean, y_noisy, y_enhanced = y_clean[:min_len], y_noisy[:min_len], y_enhanced[:min_len]

        y_aligned = align_audio(y_clean, y_enhanced)
        sisdr = calculate_sisdr(y_clean, y_aligned)
        
        # Plot
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        plt.title(f"Waveform Alignment (SI-SDR: {sisdr:.2f} dB)", fontsize=14, fontweight='bold')
        plt.plot(y_clean, 'k', alpha=0.6, label='Clean Ref')
        plt.plot(y_aligned, 'g--', alpha=0.8, label='Enhanced (Aligned)')
        plt.legend()

        plt.subplot(3, 2, 3)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_noisy)), ref=np.max), sr=sr, x_axis='time', y_axis='hz')
        plt.title("Noisy Input")
        
        plt.subplot(3, 2, 4)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_clean)), ref=np.max), sr=sr, x_axis='time', y_axis='hz')
        plt.title("Clean Target")

        plt.subplot(3, 1, 3)
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y_aligned)), ref=np.max), sr=sr, x_axis='time', y_axis='hz')
        plt.title("Enhanced Output")
        plt.tight_layout()
        plt.savefig(r"C:\Users\arjun\Desktop\hack\n100_snr0.5\final_visual_report_n50_snr0.33.png")
        plt.show()
