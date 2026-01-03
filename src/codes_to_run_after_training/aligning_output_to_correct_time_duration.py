import os
import numpy as np
import soundfile as sf
import torch
import shutil
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DIR = "/kaggle/working/enhanced_output_run2"         # Folder with your results
CLEAN_DIR = "/kaggle/input/dataset1/test/clean"        # Folder with reference audio
FINAL_OUTPUT_DIR = "/kaggle/working/enhanced_output_aligned_final"
# ---------------------

def align_and_save_torch(ref, est):
    """
    Aligns audio using PyTorch FFT to bypass Scipy errors.
    """
    # 1. Convert to Torch Tensors
    ref_t = torch.tensor(ref, dtype=torch.float32)
    est_t = torch.tensor(est, dtype=torch.float32)
    
    # 2. Calculate Cross-Correlation using FFT
    # Pad to next power of 2 for speed
    n = len(ref) + len(est) - 1
    n_fft = 2 ** (n - 1).bit_length()
    
    ref_f = torch.fft.rfft(ref_t, n=n_fft)
    est_f = torch.fft.rfft(est_t, n=n_fft)
    
    # Cross Correlation = IFFT( FFT(ref) * conj(FFT(est)) )
    cc = torch.fft.irfft(ref_f * torch.conj(est_f), n=n_fft)
    
    # 3. Find the best shift (Lag)
    max_idx = torch.argmax(cc).item()
    
    # Handle circular wrapping
    if max_idx > n_fft // 2:
        lag = max_idx - n_fft
    else:
        lag = max_idx
        
    # 4. Apply Shift
    if lag > 0:
        # est is ahead, pad start
        est_aligned = np.pad(est, (lag, 0))[:len(ref)]
    elif lag < 0:
        # est is behind, crop start
        est_aligned = est[-lag:]
        est_aligned = np.pad(est_aligned, (0, len(ref) - len(est_aligned)))
    else:
        est_aligned = est
        
    return est_aligned

# Create final folder
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

# List files
files = [f for f in os.listdir(SOURCE_DIR) if f.endswith('.wav')]
print(f"🛠️  Post-processing {len(files)} files using PyTorch Alignment...")

for file in tqdm(files):
    src_path = os.path.join(SOURCE_DIR, file)
    clean_path = os.path.join(CLEAN_DIR, file)
    dst_path = os.path.join(FINAL_OUTPUT_DIR, file)
    
    if os.path.exists(clean_path):
        # Load
        clean, sr = sf.read(clean_path)
        enhanced, _ = sf.read(src_path)
        
        # Truncate to min length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        
        # Align
        try:
            final_audio = align_and_save_torch(clean, enhanced)
            sf.write(dst_path, final_audio, sr)
        except Exception as e:
            print(f"Warning: Could not align {file} ({e}), copying original.")
            shutil.copy(src_path, dst_path)
    else:
        # If no reference, just copy the original
        shutil.copy(src_path, dst_path)

print("\n✅ Done! 'enhanced_output_aligned_final' contains your sync-corrected audio.")

# Zip the final result for download
from IPython.display import FileLink
shutil.make_archive("Final_Aligned_Results", 'zip', FINAL_OUTPUT_DIR)
print("⬇️ Download your final fixed files below:")
display(FileLink("Final_Aligned_Results.zip"))