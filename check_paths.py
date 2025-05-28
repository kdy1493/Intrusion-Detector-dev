import sys
import os
import site

# Python 실행 파일 경로
print(f"Python 실행 파일: {sys.executable}")

# Python 경로
print("\nPython 경로:")
for path in sys.path:
    print(f"  {path}")

# 사이트 패키지 경로
print("\n사이트 패키지 경로:")
for path in site.getsitepackages():
    print(f"  {path}")

# torch 모듈 확인
try:
    import torch
    print(f"\ntorch 모듈 경로: {torch.__file__}")
    print(f"torch 버전: {torch.__version__}")
except ImportError:
    print("\ntorch 모듈을 가져올 수 없습니다.") 