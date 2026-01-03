import os
import re

# Root directory to scan
SEARCH_ROOT = "/kaggle/working"

print(f"🚀 Starting TOTAL REMEDIATION SCAN in {SEARCH_ROOT}...")

# Counters
fixed_layers = 0
fixed_ncsnpp = 0

for root, dirs, files in os.walk(SEARCH_ROOT):
    
    # --- PATCH 1: layerspp.py (The current crash) ---
    if "layerspp.py" in files:
        file_path = os.path.join(root, "layerspp.py")
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            # The pattern causing the crash: self.W on CPU vs x on GPU
            # We look for: x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
            original_line = "x_proj = x[:, None] * self.W[None, :] * 2 * np.pi"
            
            # The fix: Force self.W to the device of x
            fixed_line = "x_proj = x[:, None] * self.W.to(x.device)[None, :] * 2 * np.pi"
            
            if original_line in content:
                content = content.replace(original_line, fixed_line)
                with open(file_path, "w") as f:
                    f.write(content)
                print(f"   ✅ Patched layerspp.py at: {file_path}")
                fixed_layers += 1
            elif "self.W.to(x.device)" in content:
                print(f"   🔹 Already fixed: {file_path}")
        except Exception as e:
            print(f"   ❌ Error reading {file_path}: {e}")

    # --- PATCH 2: ncsnpp.py (The likely NEXT crash) ---
    if "ncsnpp.py" in files:
        file_path = os.path.join(root, "ncsnpp.py")
        try:
            with open(file_path, "r") as f:
                content = f.read()
            
            modified = False
            
            # Fix A: The 'modules' list (middle layers)
            # Find: modules[m_idx](
            # Replace: modules[m_idx].to(x.device)(
            # We use regex to avoid double-patching
            if "modules[m_idx].to(x.device)" not in content:
                # This regex looks for 'modules[m_idx]' followed immediately by '('
                content, count = re.subn(r"modules\[m_idx\](?=\()", "modules[m_idx].to(x.device)", content)
                if count > 0: modified = True
            
            # Fix B: The specific 'temb' lines (if missed by regex)
            if "temb = modules[m_idx](temb)" in content:
                content = content.replace("temb = modules[m_idx](temb)", "temb = modules[m_idx].to(temb.device)(temb)")
                modified = True
            
            # Fix C: The output layer (final layer)
            if "h = self.output_layer(h)" in content:
                content = content.replace("h = self.output_layer(h)", "h = self.output_layer.to(h.device)(h)")
                modified = True
                
            if modified:
                with open(file_path, "w") as f:
                    f.write(content)
                print(f"   ✅ Patched ncsnpp.py at: {file_path}")
                fixed_ncsnpp += 1
            else:
                 # Check if it looks patched roughly
                if "to(x.device)" in content or "to(temb.device)" in content:
                    print(f"   🔹 Already fixed: {file_path}")

        except Exception as e:
            print(f"   ❌ Error reading {file_path}: {e}")

print("-" * 30)
print(f"🎉 Scanning Complete.")
print(f"Fixed {fixed_layers} copies of layerspp.py")
print(f"Fixed {fixed_ncsnpp} copies of ncsnpp.py")
print("You are clear to run inference again.")