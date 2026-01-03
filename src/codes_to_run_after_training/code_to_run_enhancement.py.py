import os
import subprocess

# Define paths
INPUT_DIR = "/kaggle/input/dataset1/test/noisy"  # Make sure this path is correct
OUTPUT_DIR = "/kaggle/working/enhanced_output_run2"
# Make sure the checkpoint path matches exactly where you uploaded it
CKPT_PATH = "/kaggle/input/dataset/epoch326-step408750.ckpt"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run the SGMSE+ Enhancement
# REMOVED: --backbone ncsnpp (The script loads this from the checkpoint automatically)
cmd = [
    "python", "enhancement.py",
    "--test_dir", INPUT_DIR, 
    "--enhanced_dir", OUTPUT_DIR,
    "--ckpt", CKPT_PATH,
    "--N", "50", #or 100
    "--snr", "0.33" #or 0.5
]

print("🚀 Launching Inference...")
subprocess.run(cmd)
print("Done!")