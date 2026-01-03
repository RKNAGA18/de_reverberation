# Read the enhancement.py file
file_path = "/kaggle/working/sgmse/enhancement.py"

with open(file_path, "r") as f:
    code = f.read()

# The line causing the error
old_line = "model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=args.device)"

# The fixed line that disables the security check for this specific load
# We use try/except to handle different PyTorch Lightning versions
new_line = "model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=args.device)"
# Actually, the safest hack is to monkey-patch torch.load inside the file temporarily
patch_code = """
    # --- HACKATHON HOTFIX START ---
    import torch
    # Force weights_only=False globally for this script to fix PyTorch 2.6+ error
    original_load = torch.load
    def patched_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    # --- HACKATHON HOTFIX END ---
    
    model = ScoreModel.load_from_checkpoint(args.ckpt, map_location=args.device)
"""

# We will replace the original load line with our patch block
if old_line in code:
    code = code.replace(old_line, patch_code)
    
    with open(file_path, "w") as f:
        f.write(code)
    print("✅ Successfully patched enhancement.py! You can now run the inference.")
else:
    print("⚠️ Could not find the exact line to patch. Check if the file was already modified.")