import os
import numpy as np
import soundfile as sf
import pandas as pd
from scipy import signal
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
import argparse

# --- CONFIGURATION ---
# If your folder names are different, change them here
CLEAN_DIR = r"C:\Users\arjun\Desktop\hack\SGMSE_Eval\clean"
ENHANCED_DIR = r"C:\Users\arjun\Desktop\hack\n50_snr0.33\enhanced_output_n50_snr0.33"
NOISY_DIR = r"C:\Users\arjun\Desktop\hack\SGMSE_Eval\noisy"
# ---------------------

def align_audio(ref, est):
    """
    Aligns the estimated audio (est) to the reference (ref) using cross-correlation.
    This fixes the 'diffusion latency' that causes bad SI-SDR scores.
    """
    # Calculate cross-correlation
    correlation = signal.correlate(ref, est, mode='full')
    lags = signal.correlation_lags(ref.size, est.size, mode='full')
    lag = lags[np.argmax(correlation)]

    # Apply shift
    if lag > 0:
        # est is ahead of ref (needs delay)
        est_aligned = np.pad(est, (lag, 0))[:ref.size]
    elif lag < 0:
        # est is behind ref (needs cropping)
        est_aligned = est[-lag:]
        est_aligned = np.pad(est_aligned, (0, ref.size - est_aligned.size))
    else:
        est_aligned = est
        
    return est_aligned

def calculate_sisdr(ref, est):
    """Calculates Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)"""
    eps = np.finfo(est.dtype).eps
    reference = ref.reshape(-1, 1)
    estimate = est.reshape(-1, 1)
    
    R = np.dot(reference.T, reference)
    if R == 0: return -100
    
    rho = np.dot(reference.T, estimate) / (R + eps)
    e_target = rho * reference
    e_res = estimate - e_target
    
    num = np.sum(e_target**2)
    den = np.sum(e_res**2)
    
    if den == 0: return 100
    return 10 * np.log10((num + eps) / (den + eps))

def main():
    print(f"🚀 Starting Evaluation...")
    print(f"   Clean Dir:    {CLEAN_DIR}")
    print(f"   Enhanced Dir: {ENHANCED_DIR}")

    files = [f for f in os.listdir(ENHANCED_DIR) if f.endswith('.wav')]
    results = []

    for file in tqdm(files, desc="Processing"):
        clean_path = os.path.join(CLEAN_DIR, file)
        enhanced_path = os.path.join(ENHANCED_DIR, file)
        
        # Check if reference exists
        if not os.path.exists(clean_path):
            continue

        try:
            # 1. Load Audio
            clean, sr = sf.read(clean_path)
            enhanced, _ = sf.read(enhanced_path)
            
            # 2. Length Check
            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]

            # 3. ALIGNMENT (Crucial for SI-SDR)
            enhanced_aligned = align_audio(clean, enhanced)

            # 4. Calculate Metrics
            # PESQ (Wideband)
            try:
                pesq_score = pesq(sr, clean, enhanced_aligned, 'wb')
            except:
                pesq_score = np.nan # Sometimes fails on silent audio
            
            # STOI
            stoi_score = stoi(clean, enhanced_aligned, sr, extended=False)
            
            # SI-SDR
            sisdr_score = calculate_sisdr(clean, enhanced_aligned)

            # Optional: Calculate Noisy Baseline if available
            noisy_path = os.path.join(NOISY_DIR, file)
            pesq_improv = 0
            if os.path.exists(noisy_path):
                noisy, _ = sf.read(noisy_path)
                noisy = noisy[:min_len]
                try:
                    pesq_base = pesq(sr, clean, noisy, 'wb')
                    pesq_improv = pesq_score - pesq_base
                except: pass

            results.append({
                "Filename": file,
                "PESQ": pesq_score,
                "PESQ_Improvement": pesq_improv,
                "STOI": stoi_score,
                "SI-SDR": sisdr_score
            })

        except Exception as e:
            print(f"\n⚠️ Error processing {file}: {e}")

    # --- SAVE RESULTS ---
    if results:
        df = pd.DataFrame(results)
        
        print("\n" + "="*40)
        print("📊 FINAL RESULTS SUMMARY")
        print("="*40)
        print(f"Files Evaluated: {len(df)}")
        print(f"Avg PESQ:    {df['PESQ'].mean():.4f}")
        print(f"Avg STOI:    {df['STOI'].mean():.4f}")
        print(f"Avg SI-SDR:  {df['SI-SDR'].mean():.2f} dB")
        print("="*40)
        
        df.to_csv(r"C:\Users\arjun\Desktop\hack\n50_snr0.33\metrics_report_n50_snr0.33.csv", index=False)
        print(f"✅ Detailed report saved to: metrics_report.csv")
    else:
        print("❌ No files were processed. Check your folder names!")

if __name__ == "__main__":
    main()
