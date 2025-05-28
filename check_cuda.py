import torch
import sys

print(f"Python 버전: {sys.version}")
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 버전: {torch.version.cuda}")
    print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
    print(f"CUDA 디바이스 수: {torch.cuda.device_count()}")
    
    # 모든 GPU 장치 정보 출력
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  메모리: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")
else:
    print("CUDA를 사용할 수 없습니다. CPU 모드로 동작합니다.")

# CUDA 메모리 정보 (사용 가능한 경우)
if torch.cuda.is_available():
    print("\nCUDA 메모리 정보:")
    print(f"할당된 메모리: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
    print(f"캐시된 메모리: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
    print(f"최대 메모리 캐시: {torch.cuda.max_memory_reserved() / 1024 / 1024:.2f} MB") 