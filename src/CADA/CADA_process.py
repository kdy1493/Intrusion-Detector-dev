"""
CADA_process.py
----
Wi-Fi CSI 기반 CADA( CSI Activity Detection Algorithm ) 전처리·활동 탐지 모듈.

주요 기능
----
• CSI 데이터 파싱 및 전처리
• Z-score 정규화
• CADA 알고리즘 파이프라인
• 슬라이딩 윈도우 기반 활동 탐지
• 캘리브레이션 데이터 관리
• 배경 데이터 수집 및 보정
"""

import autorootcwd
from scipy.signal import medfilt
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import re
from datetime import datetime
import os
import csv
import time
from typing import List, Dict, Tuple, Optional

# ------------------------------------------------------------------
# CSI 데이터 파싱 및 전처리
# ------------------------------------------------------------------

def parse_csi_data(payload: str, subcarriers: int, indices_to_remove: list[int] | None) -> tuple[np.ndarray, datetime] | None:
    """CSI 데이터 페이로드를 파싱하고 노이즈 채널을 제거한다.
    
    Args:
        payload: MQTT 페이로드 문자열
        subcarriers: 서브캐리어 수
        indices_to_remove: 제거할 노이즈 채널 인덱스
    
    Returns:
        (csi_amplitude, timestamp) 또는 오류 시 None
    """
    try:
        # 1) 타임스탬프 파싱
        match = re.search(r"time=(\d{15})", payload)
        if match:
            ts_str = match.group(1)
            # ESP 포맷(YYMMDDhhmmssSSS) 적용
            year = 2000 + int(ts_str[0:2])
            month = int(ts_str[2:4])
            day = int(ts_str[4:6])
            hour = int(ts_str[6:8])
            minute = int(ts_str[8:10])
            second = int(ts_str[10:12])
            millisecond = int(ts_str[12:15])
            microsecond = millisecond * 1000
            timestamp = datetime(year, month, day, hour, minute, second, microsecond)
        else:
            timestamp = datetime.now()

        # 2) CSI 문자열 → 복소수 배열
        csi_data_str = payload.split("CSI values: ")[-1].strip()
        csi_values = list(map(int, csi_data_str.split()))
        if len(csi_values) < subcarriers * 2:
            return None  # 데이터 부족

        csi_complex = [csi_values[i] + 1j * csi_values[i + 1]
                      for i in range(0, len(csi_values), 2)]
        csi_complex = np.array(csi_complex)[:subcarriers]

        # 3) 노이즈 채널 제거
        if indices_to_remove:
            csi_complex = np.delete(csi_complex, indices_to_remove)

        # 4) 진폭 계산
        csi_amplitude = np.abs(csi_complex)
        return csi_amplitude, timestamp

    except Exception as e:
        print(f"ERROR: Failed to parse CSI data: {e}")
        return None

def z_normalization(amp: np.ndarray, topic: str, mu_bg_dict: dict, sigma_bg_dict: dict) -> np.ndarray:
    """CSI 진폭 데이터를 Z-score 정규화한다.
    
    Args:
        amp: CSI 진폭 데이터
        topic: MQTT 토픽
        mu_bg_dict: 각 토픽별 평균값 딕셔너리
        sigma_bg_dict: 각 토픽별 표준편차 딕셔너리
    
    Returns:
        정규화된 진폭 데이터
    """
    if topic in mu_bg_dict and topic in sigma_bg_dict:
        return (amp - mu_bg_dict[topic]) / sigma_bg_dict[topic]
    return amp  # 캘리브레이션 데이터가 없으면 원본 반환

def filter_znormalization(amp_normalized, iqr_multiplier=1.5, gap_threshold=0.2):
    ''' 
    Desc : 
        Z-score 정규화 이후 서브캐리어 평균 기반(사분위수 사용) 이상치 제거 (단 1개만 확실히 튈 때만)
    '''
    # 1. 평균 계산 
    means = np.mean(amp_normalized, axis=0)
    # 2. IQR 계산
    q1 = np.percentile(means, 25) # 하위 25% : -2.11
    q3 = np.percentile(means, 75) # 상위 75% : 0.44
    iqr = q3 - q1
    upper = q3 + iqr_multiplier * iqr # 4.27
    # 3. 평균값 내림차순 정렬
    sorted_indices = np.argsort(means)[::-1]
    top1_idx = sorted_indices[0] # 4번
    top2_idx = sorted_indices[1] # 3번
    top1_val = means[top1_idx]  # 4번 평균 :13.42 
    top2_val = means[top2_idx]  # 3번 평균 :4.85 
    # 4. 최대 평균이 upper보다 크고, 다음 값과 차이가 충분히 날 경우만 제거
    if top1_val > upper and (top1_val - top2_val) > gap_threshold:
        invalid_indices = [top1_idx]
    else:
        invalid_indices = []
    amp_norm_filtered = np.delete(amp_normalized, invalid_indices, axis=1 )
    print(f"[filter_znormalization] Q1 = {q1:.2f}, IQR = {iqr:.2f}, upper = {upper:.2f}")
    print(f"[filter_znormalization] Top1: SC {top1_idx}, mean = {top1_val:.2f}")
    print(f"[filter_znormalization] Top2: SC {top2_idx}, mean = {top2_val:.2f}")
    print(f"[filter_znormalization] Removed: {invalid_indices}")
    return  amp_norm_filtered

def robust_hampel(col, window=5, n_sigma=3):
    """Hampel filter로 이상치 제거"""
    median = medfilt(col, kernel_size=window)
    dev    = np.abs(col - median)
    mad    = np.median(dev)
    out    = dev > n_sigma * mad
    col[out] = median[out]
    return col

def detrending_amp(amp, mean_buffer, historical_window=100):
    """
    Desc:
        2단계 Detrending을 수행하는 함수 (CSI_To_CSV.py 방식)
        - 1단계: 프레임별 평균 제거 (프레임 중심화)
        - 2단계: 서브캐리어 기준선 제거 (현재 평균 + 과거 평균의 평균)
    Parameters:
        amp : Hampel_filtered 데이터
        mean_buffer : 과거 평균값들을 저장하는 deque 버퍼
        historical_window : 기준선 계산에 사용할 초기 프레임 수 (기본값: 100)
    Returns:
        detrended : 최종 detrended 데이터
        mean_buffer : 업데이트된 평균 버퍼
    Example:
        mean_buffer = deque(maxlen=100)
        detrended, mean_buffer = detrending_amp(Hampel_filtered, mean_buffer)
    """
    # 1단계: 프레임별 평균 제거 (CSI_To_CSV.py 방식)
    mean_per_packet = np.mean(amp, axis=1, keepdims=True)
    detrended_packet = amp - mean_per_packet
    
    # 2단계: 기준선 제거 (CSI_To_CSV.py 방식)
    mean_current = np.mean(amp, axis=0)
    
    # 과거 평균값이 버퍼에 있으면 사용 (CSI_To_CSV.py 방식)
    if len(mean_buffer) > 0:
        hist_array = np.array(mean_buffer)
        mean_historical = np.mean(hist_array, axis=0)
    else:
        mean_historical = mean_current
    
    combined_mean = (mean_current + mean_historical) / 2
    
    # 최종 detrending
    detrended = detrended_packet - combined_mean
    
    # 현재 평균을 버퍼에 저장 (실시간 상태 유지)
    mean_buffer.append(mean_current.copy())
    
    return detrended, mean_buffer

def extract_motion_features(detrended, prev_samples, WIN_SIZE=64):
    """
    Desc:
        CSI detrended 데이터로부터 움직임 특성을 추출하는 함수 (CSI_To_CSV.py 방식)
        Overlap-save convolution을 사용하여 이전 상태를 유지하면서 처리
    Parameters:
        detrended : Detrended amplitude 데이터 (frames x subcarriers)
        prev_samples : 이전 프레임 정보를 저장한 배열 (WIN_SIZE 길이)
        WIN_SIZE : 이동 평균 필터 크기 (기본값: 64)
    Returns:
        feature : 최종 움직임 특징
        prev_samples : 업데이트된 이전 샘플 배열
    Example:
        prev_samples = np.zeros(WIN_SIZE)
        feature, prev_samples = extract_motion_features(detrended, prev_samples, WIN_SIZE=64)
    """
    # 1단계: 진폭 변화량 계산 (프레임별 표준편차) - CSI_To_CSV.py 방식
    SCAbsEuclidSumFeatured = np.std(detrended, axis=1)
    
    # 2단계: 변화량 미분 후 절댓값 처리 - CSI_To_CSV.py 방식
    FeaturedDerivative = np.diff(SCAbsEuclidSumFeatured)
    FeaturedDerivativeAbs = np.abs(FeaturedDerivative)
    
    # 3단계: Overlap-save convolution 기반 이동 평균 필터 - CSI_To_CSV.py 방식
    padded_signal = np.concatenate([prev_samples, FeaturedDerivativeAbs])
    window = np.ones(WIN_SIZE)
    convolved = np.convolve(padded_signal, window, mode='valid')
    
    # 4단계: 이전 샘플 업데이트 (실시간 상태 유지) - CSI_To_CSV.py 방식
    prev_samples = FeaturedDerivativeAbs[-WIN_SIZE:] if len(FeaturedDerivativeAbs) >= WIN_SIZE else np.concatenate([prev_samples[len(FeaturedDerivativeAbs):], FeaturedDerivativeAbs])
    
    # 5단계: 최신 변화량만 반환 - CSI_To_CSV.py 방식
    feature = convolved[-len(FeaturedDerivativeAbs):]
    
    return feature, prev_samples

def detect_activity_with_ewma(feature: np.ndarray,
                            ewma_state: float,
                            threshold_factor: float = 2.5,
                            alpha: float = 0.01) -> tuple[np.ndarray, float, float]:
    """EWMA를 사용하여 활동을 감지한다.
    
    Args:
        feature: 움직임 특징 데이터
        ewma_state: 이전 EWMA 상태값
        threshold_factor: 평균 대비 임계 배율
        alpha: EWMA 가중치
    
    Returns:
        activity_flag_array: 활동 감지 플래그 배열
        threshold: 현재 임계값
        ewma_state: 업데이트된 EWMA 상태
    """
    # 현재 변화량의 평균 계산 - CSI_To_CSV.py 방식
    avgSigVal = np.mean(feature) if len(feature) > 0 else 0.0
    
    # EWMA 업데이트 (실시간 상태 유지) - CSI_To_CSV.py 방식
    if ewma_state == 0.0:
        ewma_state = avgSigVal
    else:
        ewma_state = alpha * avgSigVal + (1 - alpha) * ewma_state
    
    # 임계값 계산 - CSI_To_CSV.py 방식
    threshold = threshold_factor * ewma_state
    
    # 활동 감지 - CSI_To_CSV.py 방식
    activity_flag = (feature > threshold).astype(float)
    
    return activity_flag, threshold, ewma_state

def cada_pipeline(
    amp_window: np.ndarray,
    topic: str,
    mean_buffer: deque,
    prev_samples: np.ndarray,
    ewma_state: float,
    historical_window: int = 100,
    win_size: int = 64,
    threshold_factor: float = 2.5,
    alpha: float = 0.01
) -> tuple[np.ndarray, np.ndarray, np.ndarray, deque, np.ndarray, float]:
    """
    이미 정규화된 2차원 CSI 진폭 배열(amp_window)에 대해 CADA 후처리만 수행하고,
    stride 구간의 각 프레임별 activity_detection, activity_flag, threshold 시계열 전체를 반환.

    Returns:
        features: 움직임 특징 시계열 (np.ndarray)
        activity_flags: 활동 감지 플래그 시계열 (np.ndarray)
        thresholds: threshold 시계열 (np.ndarray)
        mean_buffer: 업데이트된 평균 버퍼
        prev_samples: 업데이트된 이전 샘플 배열
        ewma_state: 업데이트된 EWMA 상태
    """
    try:
        # 1. Hampel 필터로 이상치 제거
        amp_filtered = np.apply_along_axis(robust_hampel, 0, amp_window)
        # 2. Detrending (기준선 제거)
        amp_detrended, mean_buffer = detrending_amp(amp_filtered, mean_buffer, historical_window)
        # 3. 움직임 특징 추출
        features, prev_samples = extract_motion_features(amp_detrended, prev_samples, win_size)
        # 4. 활동 감지 (프레임별)
        activity_flags = []
        thresholds = []
        ewma = ewma_state
        for f in features:
            ewma = alpha * f + (1 - alpha) * ewma
            th = threshold_factor * ewma
            flag = 1.0 if f > th else 0.0
            activity_flags.append(flag)
            thresholds.append(th)
        activity_flags = np.array(activity_flags)
        thresholds = np.array(thresholds)
        return features, activity_flags, thresholds, mean_buffer, prev_samples, ewma
    except Exception as e:
        print(f"Error in cada_pipeline: {e}")
        return np.zeros(1), np.zeros(1), np.zeros(1), mean_buffer, prev_samples, ewma_state

def load_calibration_data(topics: List[str], mu_bg_dict: Dict[str, np.ndarray], sigma_bg_dict: Dict[str, np.ndarray]) -> None:
    """
    캘리브레이션 데이터 로드 함수
    
    Parameters
    ----------
    topics : List[str]
        MQTT 토픽 목록
    mu_bg_dict : Dict[str, np.ndarray]
        배경 평균 저장 딕셔너리
    sigma_bg_dict : Dict[str, np.ndarray]
        배경 표준편차 저장 딕셔너리
    """
    try:
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

# __all__ 업데이트
globals()["__all__"] = [name for name in globals().get("__all__", []) if "Atif" not in name]
if "SlidingCadaProcessor" not in globals()["__all__"]:
    globals()["__all__"].append("SlidingCadaProcessor")

# 캘리브레이션 상수
CALIBRATION_SAMPLES = 10 * 60 * 100  # 10분 동안 정지상태 데이터 수집
CALIB_DIR = "data/calibration"
os.makedirs(CALIB_DIR, exist_ok=True)

def collect_calibration_data(topic: str, csi_data: np.ndarray, calibration_buffer: List[np.ndarray], 
                           target_samples: int = CALIBRATION_SAMPLES) -> bool:
    """
    캘리브레이션 데이터 수집 함수
    
    Parameters
    ----------
    topic : str
        MQTT 토픽
    csi_data : np.ndarray
        CSI 데이터
    calibration_buffer : List[np.ndarray]
        캘리브레이션 데이터 버퍼
    target_samples : int
        목표 샘플 수
    
    Returns
    -------
    bool
        캘리브레이션 완료 여부
    """
    calibration_buffer.append(csi_data)
    if len(calibration_buffer) >= target_samples:
        print(f"Collected {len(calibration_buffer)} samples for {topic}")
        return True
    return False

def calibrate_background(topic: str, calibration_data: np.ndarray, force_new: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    배경 데이터 보정 함수
    
    Parameters
    ----------
    topic : str
        MQTT 토픽
    calibration_data : np.ndarray
        캘리브레이션 데이터
    force_new : bool
        강제로 새로운 캘리브레이션 수행 여부
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (평균, 표준편차) 튜플
    """
    calib_file = os.path.join(CALIB_DIR, f"{topic.replace('/','_')}_bg_params.csv")
    
    if not force_new and os.path.exists(calib_file):
        # 기존 캘리브레이션 데이터 로드
        with open(calib_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        mu_bg = np.array([float(x) for x in rows[0]])
        sigma_bg = np.array([float(x) for x in rows[1]])
        print(f"[{topic}] Loaded: {calib_file}")
    else:
        # 새로운 캘리브레이션 수행
        mu_bg = np.mean(calibration_data, axis=0)
        sigma_bg = np.std(calibration_data, axis=0)
        sigma_bg[sigma_bg == 0] = 1  # 표준편차가 0인 경우 1로 처리
        
        # 캘리브레이션 데이터 저장
        with open(calib_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(mu_bg)
            writer.writerow(sigma_bg)
        print(f"[{topic}] Saved: {calib_file}")
    
    return mu_bg, sigma_bg

def run_parallel_calibration(topics: List[str], force_new: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    병렬 캘리브레이션 실행 함수
    
    Parameters
    ----------
    topics : List[str]
        MQTT 토픽 목록
    force_new : bool
        강제로 새로운 캘리브레이션 수행 여부
    
    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        (평균 딕셔너리, 표준편차 딕셔너리) 튜플
    """
    mu_bg_dict: Dict[str, np.ndarray] = {}
    sigma_bg_dict: Dict[str, np.ndarray] = {}
    
    if force_new:
        # 새로운 캘리브레이션 수행
        calibration_buffers = {topic: [] for topic in topics}
        calibration_done = {topic: False for topic in topics}
        
        print("Collecting No Movement Data...")
        while not all(calibration_done.values()):
            for topic in topics:
                if not calibration_done[topic]:
                    # 여기서 CSI 데이터를 받아와야 함
                    # 실제 구현에서는 MQTT 클라이언트나 다른 데이터 소스에서 데이터를 받아와야 함
                    pass
            time.sleep(0.01)
        
        # 캘리브레이션 데이터 처리
        for topic in topics:
            calibration_data = np.array(calibration_buffers[topic])
            mu_bg, sigma_bg = calibrate_background(topic, calibration_data, force_new=True)
            mu_bg_dict[topic] = mu_bg
            sigma_bg_dict[topic] = sigma_bg
        
        print("Calibration complete for all topics.")
    else:
        # 기존 캘리브레이션 데이터 로드
        for topic in topics:
            dummy = np.zeros((1, 52 - len(range(21, 32))))  # 서브캐리어 수에 맞게 조정
            mu_bg, sigma_bg = calibrate_background(topic, dummy, force_new=False)
            mu_bg_dict[topic] = mu_bg
            sigma_bg_dict[topic] = sigma_bg
            print(f"Loaded calibration for {topic}...")
        
        print("Calibration loaded.")
    
    return mu_bg_dict, sigma_bg_dict