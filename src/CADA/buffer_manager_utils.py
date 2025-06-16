"""
buffer_manager_utils.py
------
Wi-Fi CSI 실시간 데이터 버퍼 관리 모듈

기능
------
• 실시간 CSI 데이터 버퍼 관리
• 타임스탬프 버퍼 관리
• CADA 특징 버퍼 관리
"""

import autorootcwd
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Deque, Any

class BufferManager:
    """실시간 CSI 데이터 버퍼 관리 클래스"""
    
    def __init__(self, topics: List[str], buffer_size: int = 512, window_size: int = 64):
        """
        Parameters
        ----------
        topics : List[str]
            MQTT 토픽 목록
        buffer_size : int
            버퍼 최대 크기
        window_size : int
            윈도우 크기
        """
        self.topics = topics
        self.buffer_size = buffer_size
        self.window_size = window_size
        
        # CADA용 배경 평균/표준편차 저장용 딕셔너리 및 EWMA 상태값 초기화
        self.mu_bg_dict: Dict[str, np.ndarray] = {topic: np.array([]) for topic in self.topics}
        self.sigma_bg_dict: Dict[str, np.ndarray] = {topic: np.array([]) for topic in self.topics}
        self.cada_ewma_states: Dict[str, float] = {topic: 0.0 for topic in self.topics}
        
        # 버퍼 초기화
        self._init_buffers()
    
    def _init_buffers(self) -> None:
        """모든 버퍼 초기화"""
        # 타임스탬프 버퍼
        self.timestamp_buffer: Dict[str, Deque[datetime]] = {
            topic: deque(maxlen=self.buffer_size) for topic in self.topics
        }
        
        # CSI 데이터 버퍼
        self.csi_buffers: Dict[str, Deque[np.ndarray]] = {
            topic: deque(maxlen=self.buffer_size) for topic in self.topics
        }
        
        # CADA 특징 버퍼
        self.feature_buffers: Dict[str, Dict[str, Deque[float]]] = {
            'activity_detection': {topic: deque(maxlen=self.buffer_size) for topic in self.topics},
            'activity_flag': {topic: deque(maxlen=self.buffer_size) for topic in self.topics},
            'threshold': {topic: deque(maxlen=self.buffer_size) for topic in self.topics}
        }
        
        # CADA 상태 버퍼
        self.mean_buffers: Dict[str, Deque[np.ndarray]] = {
            topic: deque(maxlen=100) for topic in self.topics
        }
        self.prev_samples: Dict[str, np.ndarray] = {
            topic: np.zeros(self.window_size) for topic in self.topics
        }
        self.ewma_states: Dict[str, float] = {
            topic: 0.0 for topic in self.topics
        }
    
    def get_features(self) -> Dict[str, Dict[str, Deque[float]]]:
        """CADA 특징 딕셔너리 반환"""
        return self.feature_buffers
    
    def clear(self) -> None:
        """모든 버퍼 초기화"""
        for topic in self.topics:
            # 타임스탬프와 CSI 버퍼 초기화
            self.timestamp_buffer[topic].clear()
            self.csi_buffers[topic].clear()
            
            # 특징 버퍼 초기화
            for feat_buffer in self.feature_buffers.values():
                feat_buffer[topic].clear()
            
            # 상태 버퍼 초기화
            self.mean_buffers[topic].clear()
            self.prev_samples[topic] = np.zeros(self.window_size)
            self.ewma_states[topic] = 0.0
    
    def add_csi_data(self, topic: str, csi_data: np.ndarray, timestamp: datetime) -> None:
        """CSI 데이터 추가"""
        self.csi_buffers[topic].append(csi_data)
        self.timestamp_buffer[topic].append(timestamp)
    
    def add_feature(self, topic: str, feature_name: str, value: float) -> None:
        """CADA 특징 추가"""
        if feature_name in self.feature_buffers:
            self.feature_buffers[feature_name][topic].append(value)
    
    def get_latest_csi(self, topic: str) -> np.ndarray:
        """최신 CSI 데이터 반환"""
        return np.array(list(self.csi_buffers[topic])[-self.window_size:])
    
    def get_latest_features(self, topic: str) -> Dict[str, float]:
        """최신 CADA 특징 반환"""
        return {
            name: buffer[topic][-1] if buffer[topic] else 0.0
            for name, buffer in self.feature_buffers.items()
        }

def create_buffer_manager(topics: List[str], buffer_size: int = 512, window_size: int = 64) -> BufferManager:
    """버퍼 매니저 생성 함수"""
    return BufferManager(topics, buffer_size, window_size) 

# ---------------------------------------------------------------
# 외부 호환성을 위한 래퍼 함수
# demo/app.py 및 기타 모듈에서
# `from src.CADA.buffer_manager_utils import load_calibration_data`
# 로 임포트하는 경우가 있어 이를 재노출한다.
# ---------------------------------------------------------------
from .CADA_process import load_calibration_data  # type: ignore 