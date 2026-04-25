import torch
import sys


def main() -> None:
    print("=" * 50)
    print("      Project Environment Verification")
    print("=" * 50)
    
    # 1. 현재 실행 중인 Python과 PyTorch 버전을 먼저 출력합니다.
    #    CUDA 문제는 라이브러리 조합 차이에서 시작되는 경우가 많아 가장 먼저 확인합니다.
    print(f"[*] Python Version  : {sys.version.split()[0]}")
    print(f"[*] PyTorch Version : {torch.__version__}")
    
    # 2. CUDA 사용 가능 여부와 장치 정보를 확인합니다.
    cuda_available = torch.cuda.is_available()
    print(f"[*] CUDA Available  : {cuda_available}")
    
    if cuda_available:
        print(f"[*] GPU Device Name : {torch.cuda.get_device_name(0)}")
        print(f"[*] CUDA Capability : {torch.cuda.get_arch_list()}")
        
        # 3. 마지막으로 실제 CUDA 텐서 연산이 되는지 검증합니다.
        #    장치가 보이더라도 텐서 이동이 실패하면 실행 환경은 아직 정상이라고 볼 수 없습니다.
        try:
            torch.ones(1).cuda()
            print("[*] GPU Operation   : SUCCESS (Tensor moved to CUDA)")
        except Exception as error:
            print(f"[!] GPU Operation   : FAILED ({error})")
    else:
        print("[!] GPU Operation   : SKIPPED (CUDA not available)")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
