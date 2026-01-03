
import os
import glob
import pandas as pd
from speechmos import dnsmos
from tqdm import tqdm

# --- CONFIGURATION ---
INPUT_FOLDER = r"C:\Users\arjun\Desktop\hack\n100_snr0.5\enhanced_output_n100_snr0.5"
OUTPUT_CSV = r"C:\Users\arjun\Desktop\hack\n50_snr0.33\dnsmos_report_n100_snr0.5"
# ---------------------

def get_dnsmos_folder(folder_path):
    if not os.path.exists(folder_path):
        return pd.DataFrame()

    files = glob.glob(os.path.join(folder_path, "*.wav"))
    results = []
    
    print(f"🚀 Processing {len(files)} files...")

    for file_path in tqdm(files):
        try:
            scores = dnsmos.run(file_path, sr=16000)
            
            # Map the specific keys found in your terminal output
            results.append({
                "filename": os.path.basename(file_path),
                "DNSMOS_OVRL": float(scores.get('ovrl_mos', 0)),
                "DNSMOS_SIG": float(scores.get('sig_mos', 0)),
                "DNSMOS_BAK": float(scores.get('bak_mos', 0)),
                "P808_MOS": float(scores.get('p808_mos', 0))
            })
        except Exception as e:
            print(f"⚠️ Error on {os.path.basename(file_path)}: {e}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    df_scores = get_dnsmos_folder(INPUT_FOLDER)
    
    if not df_scores.empty:
        print("\n" + "="*40)
        print("📊 FINAL DNSMOS SUMMARY")
        print("="*40)
        # We calculate means while ignoring any zeros from errors
        avg_ovrl = df_scores["DNSMOS_OVRL"].mean()
        avg_sig = df_scores["DNSMOS_SIG"].mean()
        avg_bak = df_scores["DNSMOS_BAK"].mean()
        
        print(f"Avg Overall (OVRL): {avg_ovrl:.3f} / 5.0")
        print(f"Avg Signal (SIG):   {avg_sig:.3f} / 5.0")
        print(f"Avg Background (BAK):{avg_bak:.3f} / 5.0")
        print("="*40)
        
        df_scores.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Success! Report saved to: {OUTPUT_CSV}")