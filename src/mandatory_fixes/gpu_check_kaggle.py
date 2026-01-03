import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("✅ SUCCESS: You are ready to run the enhancement.")
else:
    print("❌ ERROR: Still running on CPU. Did you switch the Accelerator to T4?")