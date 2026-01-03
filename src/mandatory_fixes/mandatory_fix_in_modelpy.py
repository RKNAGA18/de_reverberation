import os
import re

file_path = "/kaggle/working/sgmse/sgmse/model.py"

print(f"🔧 Starting Comprehensive Fixes for: {file_path}")

if os.path.exists(file_path):
    # 1. Read the file content
    with open(file_path, "r") as f:
        content = f.read()
    
    # =========================================================================
    # FIX 1: Distributed Data Parallel (DDP) Safety
    # =========================================================================
    # Problem: The original code assumes Multi-GPU execution (using dist.get_rank()).
    #          On a Single GPU (like Kaggle default), these calls crash because 
    #          the process group is not initialized.
    # Solution: We replace them with safe versions that check initialization first.

    # Fix 'get_rank' -> Defaults to 0 if not initialized
    if "dist.get_rank()" in content:
        content = re.sub(
            r"dist\.get_rank\(\)", 
            "(dist.get_rank() if dist.is_initialized() else 0)", 
            content
        )
        print("    ✅ Fixed 'dist.get_rank()' calls (DDP Safety).")

    # Fix 'get_world_size' -> Defaults to 1 if not initialized
    if "dist.get_world_size()" in content:
        content = re.sub(
            r"dist\.get_world_size\(\)", 
            "(dist.get_world_size() if dist.is_initialized() else 1)", 
            content
        )
        print("    ✅ Fixed 'dist.get_world_size()' calls (DDP Safety).")

    # =========================================================================
    # FIX 2: PyTorch Method Signature Mismatch
    # =========================================================================
    # Problem: The custom 'train' method is defined as: def train(self, mode, ...)
    #          It is missing a default value for 'mode'.
    #          PyTorch internals often call model.train() with NO arguments, 
    #          which causes a "missing required argument" TypeError.
    # Solution: We add the standard default: mode=True.

    original_sig = "def train(self, mode, no_ema=False):"
    fixed_sig    = "def train(self, mode=True, no_ema=False):"

    if original_sig in content:
        content = content.replace(original_sig, fixed_sig)
        print("    ✅ Fixed 'train()' method signature (API Mismatch).")
    else:
        # Check if it was possibly already fixed or formatted differently
        if fixed_sig not in content:
             print("    ⚠️ 'train()' signature pattern not found. Check if file is already patched.")

    # =========================================================================
    # 3. Write back the changes
    # =========================================================================
    with open(file_path, "w") as f:
        f.write(content)
        
    print("\n🎉 Model file is now fully patched and safe for Single GPU training!")

else:
    print("❌ File not found. Make sure you are in the correct directory.")
