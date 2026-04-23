import torch
import sys


def main() -> None:
    print("=" * 50)
    print("      Project Environment Verification")
    print("=" * 50)
    
    # 1. Python & PyTorch Version
    print(f"[*] Python Version  : {sys.version.split()[0]}")
    print(f"[*] PyTorch Version : {torch.__version__}")
    
    # 2. CUDA Status
    cuda_available = torch.cuda.is_available()
    print(f"[*] CUDA Available  : {cuda_available}")
    
    if cuda_available:
        print(f"[*] GPU Device Name : {torch.cuda.get_device_name(0)}")
        print(f"[*] CUDA Capability : {torch.cuda.get_arch_list()}")
        
        # 3. Simple Tensor Operation Check
        try:
            x = torch.ones(1).cuda()
            print("[*] GPU Operation   : SUCCESS (Tensor moved to CUDA)")
        except Exception as e:
            print(f"[!] GPU Operation   : FAILED ({e})")
    else:
        print("[!] GPU Operation   : SKIPPED (CUDA not available)")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
