"""
realtime_csi_handler_utils.py
----
Wi-Fi CSI 실시간 데이터 처리·버퍼 관리 유틸리티 모듈.

주요 기능
+----
Wi-Fi CSI 실시간 데이터 처리·버퍼 관리 유틸리티 모듈.

주요 기능
+----
• RealtimeBufferManager 클래스: 실시간 CSI·특징 버퍼 관리 기능.
• create_buffer_manager 함수: 버퍼 매니저 생성 편의 기능.
• process_realtime_csi 함수: MQTT CSI 페이로드 실시간 처리 기능.
• extract_cada_features 함수: CADA 특징 추출 기능.
• load_calibration_data 함수: 캘리브레이션 CSV 로드 기능.
• parse_custom_timestamp 함수: ESP 커스텀 타임스탬프 변환 기능.
"""

import autorootcwd
import numpy as np
import re
from datetime import datetime
from collections import deque
# CADA 버전 파이프라인 및 함수 임포트는 지연 로딩 방식으로 함수 내부에서 수행한다.

class RealtimeBufferManager:
    """실시간 CSI 처리를 위한 모든 버퍼들을 관리하는 클래스"""
    
    def __init__(self, topics, buffer_size=512, window_size=64):
        """
        Parameters:
            topics : MQTT 토픽 리스트
            buffer_size : 버퍼 최대 크기
            window_size : 윈도우 크기
        """
        self.topics = topics
        self.buffer_size = buffer_size
        self.window_size = window_size
        
        # 기본 버퍼들
        self.timestamp_buffer = {topic: deque(maxlen=buffer_size) for topic in topics}
        
        # CADA 활동 감지 관련 버퍼들
        self.cada_csi_buffers = {topic: deque(maxlen=buffer_size) for topic in topics}
        self.cada_feature_buffers = {
            'activity_detection': {topic: deque(maxlen=buffer_size) for topic in topics},
            'activity_flag': {topic: deque(maxlen=buffer_size) for topic in topics},
            'threshold': {topic: deque(maxlen=buffer_size) for topic in topics}
        }
        
        # CADA 상태 변수들
        self.cada_mean_buffers = {topic: deque(maxlen=100) for topic in topics}
        self.cada_prev_samples = {topic: np.zeros(window_size) for topic in topics}
        self.cada_ewma_states = {topic: 0.0 for topic in topics}
        
        # 캘리브레이션 데이터
        self.mu_bg_dict = {}
        self.sigma_bg_dict = {}
    
    def get_combined_features(self):
        """CADA 활동 감지 특징 딕셔너리 반환"""
        combined = {}
        
        # CADA 특징들 추가
        for feat_name, feat_buffer in self.cada_feature_buffers.items():
            combined[feat_name] = feat_buffer
        
        return combined
    
    def clear_all_buffers(self):
        """모든 버퍼 초기화"""
        for topic in self.topics:
            self.timestamp_buffer[topic].clear()
            self.cada_csi_buffers[topic].clear()
            
            for feat_buffer in self.cada_feature_buffers.values():
                feat_buffer[topic].clear()
            
            self.cada_mean_buffers[topic].clear()
            self.cada_prev_samples[topic] = np.zeros(self.window_size)
            self.cada_ewma_states[topic] = 0.0

def create_buffer_manager(topics, buffer_size=512, window_size=64):
    """간편한 버퍼 매니저 생성 함수"""
    return RealtimeBufferManager(topics, buffer_size, window_size)

def process_realtime_csi(topic, payload, 
                        timestamp_buffer, 
                        cada_csi_buffers, cada_feature_buffers,
                        cada_mean_buffers, cada_prev_samples, cada_ewma_states,
                        mu_bg_dict, sigma_bg_dict,
                        beamforming_weights,
                        subcarriers=52, indices_to_remove=None, 
                        top_k=5, window_size=64, buffer_size=512,
                        pca_csi_buffers=None, pca_feature_buffers=None, pca_model=None, pca_scaler=None):
    """
    실시간 CSI 데이터 처리 함수
    
    Parameters:
        topic : MQTT 토픽
        payload : MQTT 페이로드
        timestamp_buffer : 타임스탬프 버퍼 딕셔너리
        
        cada_csi_buffers : CADA CSI 버퍼 딕셔너리
        cada_feature_buffers : CADA 특징 버퍼 딕셔너리
        cada_mean_buffers : CADA 평균 버퍼 딕셔너리
        cada_prev_samples : CADA 이전 샘플 딕셔너리
        cada_ewma_states : CADA EWMA 상태 딕셔너리
        
        mu_bg_dict : 배경 평균 딕셔너리
        sigma_bg_dict : 배경 표준편차 딕셔너리
        beamforming_weights : 빔포밍 가중치 행렬
        subcarriers : 서브캐리어 개수
        indices_to_remove : 제거할 채널 인덱스 리스트
        top_k : 상위 k개 채널 사용
        window_size : 윈도우 크기
        buffer_size : 버퍼 크기
        
        pca_csi_buffers : PCA CSI 버퍼 딕셔너리 (선택사항)
        pca_feature_buffers : PCA 특징 버퍼 딕셔너리 (선택사항)
        pca_model : PCA 모델 (선택사항)
        pca_scaler : PCA scaler (선택사항)
    """

    
    try:
        # 타임스탬프 추출
        match = re.search(r'time=(\d{15})', payload)
        if match:
            ts_str = match.group(1)
            packet_time = parse_custom_timestamp(ts_str)
        else:
            packet_time = datetime.now()
        
        # CSI 데이터 파싱
        csi_data_str = payload.split("CSI values: ")[1].strip()
        csi_values = list(map(int, csi_data_str.split()))
        
        if len(csi_values) < subcarriers * 2:
            return
            
        # 복소수 변환
        csi_complex = np.array([csi_values[i] + 1j * csi_values[i + 1] 
                               for i in range(0, len(csi_values), 2)])[:subcarriers]
        
        # 노이즈 채널 제거
        if indices_to_remove:
            csi_complex = np.delete(csi_complex, indices_to_remove)
        
        # 타임스탬프 저장
        timestamp_buffer[topic].append(packet_time)
        
        # ---- CADA 파이프라인 함수 지연 임포트 ----
        from src.CADA.CADA_process import realtime_cada_pipeline, z_normalization  # pylint: disable=import-error,cyclic-import

        # CADA 활동 감지 특징 추출
        extract_cada_features(
            topic,
            csi_complex,
            cada_csi_buffers,
            cada_feature_buffers,
            cada_mean_buffers,
            cada_prev_samples,
            cada_ewma_states,
            mu_bg_dict,
            sigma_bg_dict,
            window_size,
            realtime_cada_pipeline,
            z_normalization,
        )
        
    except Exception as e:
        print(f"Error processing CSI data for topic {topic}: {e}")

def extract_cada_features(topic, csi_complex, cada_csi_buffers, cada_feature_buffers,
                         cada_mean_buffers, cada_prev_samples, cada_ewma_states,
                         mu_bg_dict, sigma_bg_dict, window_size, realtime_cada_pipeline, z_normalization):
    """실시간 CADA 통합 파이프라인을 사용한 특징 추출"""
    try:
        # CSI 버퍼에 저장 (CADA용)
        cada_csi_buffers[topic].append(csi_complex)
        
        # 충분한 데이터가 있을 때만 처리
        if len(cada_csi_buffers[topic]) >= window_size:
            recent_data = np.array(list(cada_csi_buffers[topic])[-window_size:])
            amplitude = np.abs(recent_data)
            
            # Z-score 정규화
            if topic in mu_bg_dict and topic in sigma_bg_dict:
                amp_normalized = z_normalization(amplitude, mu_bg_dict[topic], sigma_bg_dict[topic])
            else:
                # 캘리브레이션 데이터가 없으면 간단한 정규화
                amp_normalized = (amplitude - np.mean(amplitude, axis=0)) / (np.std(amplitude, axis=0) + 1e-8)
            
            # 실시간 CADA 통합 파이프라인 실행
            activity_detection, activity_flag, threshold, updated_mean_buffer, updated_prev_samples, updated_ewma_state = \
                realtime_cada_pipeline(
                    amp_normalized, 
                    cada_mean_buffers[topic], 
                    cada_prev_samples[topic], 
                    cada_ewma_states[topic],
                    historical_window=50,  # 실시간에서는 더 작은 윈도우 사용
                    WIN_SIZE=window_size,
                    threshold_factor=2.5,
                    alpha=0.01
                )
            
            # 상태 업데이트
            cada_mean_buffers[topic] = updated_mean_buffer
            cada_prev_samples[topic] = updated_prev_samples
            cada_ewma_states[topic] = updated_ewma_state
            
            # 결과 저장
            cada_feature_buffers['activity_detection'][topic].append(activity_detection)
            cada_feature_buffers['activity_flag'][topic].append(activity_flag)
            cada_feature_buffers['threshold'][topic].append(threshold)
        else:
            # 데이터가 부족한 경우 기본값 저장
            cada_feature_buffers['activity_detection'][topic].append(0.0)
            cada_feature_buffers['activity_flag'][topic].append(0.0)
            cada_feature_buffers['threshold'][topic].append(0.1)
        
    except Exception as e:
        print(f"Error extracting CADA features for {topic}: {e}")
        # 오류 발생 시 기본값 저장
        cada_feature_buffers['activity_detection'][topic].append(0.0)
        cada_feature_buffers['activity_flag'][topic].append(0.0)
        cada_feature_buffers['threshold'][topic].append(0.1)

def load_calibration_data(topics, mu_bg_dict, sigma_bg_dict):
    """캘리브레이션 데이터 로드 함수"""
    try:
        import os
        import csv
        
        CALIB_DIR = "data/calibration"
        
        for topic in topics:
            calib_file = os.path.join(CALIB_DIR, f"{topic.replace('/','_')}_bg_params.csv")
            
            if os.path.exists(calib_file):
                with open(calib_file, 'r') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                mu_bg = np.array([float(x) for x in rows[0]])
                sigma_bg = np.array([float(x) for x in rows[1]])
                sigma_bg[sigma_bg == 0] = 1  # 0인 σ는 1로 대체
                
                mu_bg_dict[topic] = mu_bg
                sigma_bg_dict[topic] = sigma_bg
                print(f" Loaded calibration for {topic}")
            else:
                print(f" No calibration file found for {topic}: {calib_file}")
                
    except Exception as e:
        print(f"Error loading calibration data: {e}")

# ------------------------------------------------------------------
# Timestamp helper (moved from timestamp_utils.py)
# ------------------------------------------------------------------

def parse_custom_timestamp(ts):
    """ESP 15자리 타임스탬프(YYMMDDhhmmssSSS)를 datetime 으로 변환"""
    ts_str = str(ts).zfill(15)
    year = 2000 + int(ts_str[0:2])
    month = int(ts_str[2:4])
    day = int(ts_str[4:6])
    hour = int(ts_str[6:8])
    minute = int(ts_str[8:10])
    second = int(ts_str[10:12])
    millisecond = int(ts_str[12:15])
    microsecond = millisecond * 1000
    return datetime(year, month, day, hour, minute, second, microsecond)

from typing import Union  # for type hints elsewhere 