"""
CADA_process.py
----
Wi-Fi CSI 기반 CADA( CSI Activity Detection Algorithm ) 전처리·활동 탐지 모듈.

주요 기능
----
• z_normalization 함수: Z-score 정규화 기능.
• filter_normalization 함수: 정규화 후 이상치 제거 기능.
• realtime_cada_pipeline / batch_cada_pipeline 함수: 실시간·오프라인 파이프라인 기능.
• SlidingCadaProcessor 클래스: 슬라이딩 윈도우 기반 활동 탐지 기능.
• parse_and_normalize_payload 함수: MQTT 페이로드 파싱 및 Z-score 변환 기능.
"""

import autorootcwd
from scipy.signal import medfilt
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from src.CADA.plot_utils import convert_csi_to_amplitude, plot_csi_amplitude, plot_csi_amplitude_from_file
from src.CADA.realtime_csi_handler_utils import parse_custom_timestamp



def read_calibration(CALIB_PATH): # src/atif.py
    '''
    Read calibration data
    param CALIB_PATH : path to the Calibration data 
    Example :
        CALIB_PATH = r"data\CSI_Calibration\L0382_ESP_8_bg_params.csv"
        mu_bg, sigma_bg = read_calibration(CALIB_PATH)
    '''
    calib = pd.read_csv(CALIB_PATH, header=None)
    mu_bg    = calib.iloc[0].values.astype(float)
    sigma_bg = calib.iloc[1].values.astype(float) 
    sigma_bg[sigma_bg == 0] = 1  # 0인 σ는 1로 대체 (0으로 나누기 방지)
    return mu_bg, sigma_bg

def z_normalization(amp, mu, sigma ) : 
    '''
    Desc :
        Z-score normalizaion & visualization
        param mu_bg : 각 서브캐리어에 대한 평균값
        param sigma_bg : 각 서브캐리어에 대한 표준편차
    Example : 
        mu_bg, sigma_bg = read_calibration(_, _) 
        z_normalization(amp_reduced, mu_bg, sigma_bg)
    '''
    mu_bg = mu
    sigma_bg = sigma
    amp_normalized  = (amp - mu_bg) / sigma_bg
    return amp_normalized

def filter_normalization(amp_normalized, iqr_multiplier=1.5, gap_threshold=0.2):
    ''' 
    Desc : 
        normalization 이후 서브캐리어 평균 기반(사분위수 사용용) 이상치 제거 (단 1개만 확실히 튈 때만)
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
    print(f"[filter_normalization] Q1 = {q1:.2f}, IQR = {iqr:.2f}, upper = {upper:.2f}")
    print(f"[filter_normalization] Top1: SC {top1_idx}, mean = {top1_val:.2f}")
    print(f"[filter_normalization] Top2: SC {top2_idx}, mean = {top2_val:.2f}")
    print(f"[filter_normalization] Removed: {invalid_indices}")
    return  amp_norm_filtered

def robust_hampel(col, window=5, n_sigma=3):
    """Hampel filter로 이상치 제거"""
    median = medfilt(col, kernel_size=window)
    dev    = np.abs(col - median)
    mad    = np.median(dev)
    out    = dev > n_sigma * mad
    col[out] = median[out]
    return col



# === 실시간 처리를 위한 새로운 함수들 (CSI_To_CSV.py 방식 참고) ===

def realtime_detrending_amp(amp, mean_buffer, historical_window=100):
    """
    Desc:
        실시간 2단계 Detrending을 수행하는 함수 (CSI_To_CSV.py 방식)
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
        detrended, mean_buffer = realtime_detrending_amp(Hampel_filtered, mean_buffer)
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

def realtime_extract_motion_features(detrended, prev_samples, WIN_SIZE=64):
    """
    Desc:
        실시간 CSI detrended 데이터로부터 움직임 특성을 추출하는 함수 (CSI_To_CSV.py 방식)
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
        feature, prev_samples = realtime_extract_motion_features(detrended, prev_samples, WIN_SIZE=64)
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

def realtime_detect_activity_with_ewma(feature, ewma_state, threshold_factor=2.5, alpha=0.01):
    """
    Desc:
        실시간 CSI 데이터에서 EWMA 방식으로 임계선을 계산하고, 움직임 감지를 수행하는 함수 (CSI_To_CSV.py 방식)
        상태를 유지하면서 실시간으로 EWMA를 업데이트
    Parameters:
        feature : 입력된 변화량 시계열 (1D array-like)
        ewma_state : 이전 EWMA 상태값 (float)
        threshold_factor : 평균 대비 임계 배율 (기본값: 2.5)
        alpha : EWMA 가중치 (기본값: 0.01, CSI_To_CSV.py와 동일)
    Returns:
        activity_flag : 활동 감지 플래그 배열
        threshold : 현재 임계값
        ewma_state : 업데이트된 EWMA 상태
    Example:
        ewma_state = 0.0
        activity_flag, threshold, ewma_state = realtime_detect_activity_with_ewma(feature, ewma_state)
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

# ==========================================================================================
# === 실시간 처리를 위한 통합 함수 =============================================================

def realtime_cada_pipeline(amp_normalized, mean_buffer, prev_samples, ewma_state, 
                          historical_window=100, WIN_SIZE=64, threshold_factor=2.5, alpha=0.01):
    """
    Desc:
        CADA 파이프라인을 실시간 처리가 가능하도록 통합한 함수
        CSI_To_CSV.py의 상태 유지 방식을 적용
    Parameters:
        amp_normalized : Z-score 정규화된 진폭 데이터
        mean_buffer : 과거 평균값들을 저장하는 deque 버퍼
        prev_samples : 이전 프레임 정보를 저장한 배열
        ewma_state : 이전 EWMA 상태값
        historical_window : 기준선 계산에 사용할 초기 프레임 수
        WIN_SIZE : 이동 평균 필터 크기
        threshold_factor : 평균 대비 임계 배율
        alpha : EWMA 가중치
    Returns:
        activity_detection : 활동 감지 값 (평균)
        activity_flag : 활동 감지 플래그 (평균)
        threshold : 현재 임계값
        mean_buffer : 업데이트된 평균 버퍼
        prev_samples : 업데이트된 이전 샘플 배열
        ewma_state : 업데이트된 EWMA 상태
    Example:
        mean_buffer = deque(maxlen=100)
        prev_samples = np.zeros(64)
        ewma_state = 0.0
        
        activity_detection, activity_flag, threshold, mean_buffer, prev_samples, ewma_state = \
            realtime_cada_pipeline(amp_normalized, mean_buffer, prev_samples, ewma_state)
    """
    try:
        # 1. Hampel 필터 적용
        Hampel_filtered = np.apply_along_axis(robust_hampel, 0, amp_normalized)
        
        # 2. 실시간 Detrending
        detrended, mean_buffer = realtime_detrending_amp(Hampel_filtered, mean_buffer, historical_window)
        
        # 3. 실시간 움직임 특징 추출
        feature, prev_samples = realtime_extract_motion_features(detrended, prev_samples, WIN_SIZE)
        
        # 4. 실시간 활동 감지
        activity_flag_array, threshold, ewma_state = realtime_detect_activity_with_ewma(
            feature, ewma_state, threshold_factor, alpha)
        
        # 5. 결과 요약 (실시간 출력용)
        activity_detection = np.mean(feature) if len(feature) > 0 else 0.0
        activity_flag = np.mean(activity_flag_array) if len(activity_flag_array) > 0 else 0.0
        
        return activity_detection, activity_flag, threshold, mean_buffer, prev_samples, ewma_state
        
    except Exception as e:
        print(f"Error in realtime_cada_pipeline: {e}")
        return 0.0, 0.0, 0.1, mean_buffer, prev_samples, ewma_state

# === 기존 배치 처리 함수들 (하위 호환성 유지) ===
def detrending_amp(amp, historical_window=100):
    """
    Desc:
        2단계 Detrending을 수행하는 함수 (배치 처리용)
        - 1단계: 프레임별 평균 제거 (프레임 중심화)
        - 2단계: 서브캐리어 기준선 제거 (현재 평균 + 과거 평균의 평균)
    Parameters:
        amp : Hampel_filtered
        historical_window : 기준선 계산에 사용할 초기 프레임 수 (기본값: 100)
    Example:
        Hampel_filtered = np.apply_along_axis(robust_hampel, 0, amp_norm_filtered)
        detrended = detrending_amp(Hampel_filtered, historical_window=100)
    """
    # 1단계: 프레임별 평균 제거
    mean_per_frame = np.mean(amp, axis=1, keepdims=True)  
    detrended_packet = amp - mean_per_frame             
    # 2단계: 기준선 제거 (시간 평균 기준)
    mean_current = np.mean(amp, axis=0)                  
    mean_historical = np.mean(amp[:historical_window], axis=0)
    combined_mean = (mean_current + mean_historical) / 2    
    # 최종 detrending
    detrended = detrended_packet - combined_mean           
    return detrended
def extract_motion_features(detrended, WIN_SIZE=64):
    """
    Desc:
        CSI detrended 데이터로부터 움직임 특성을 추출하는 함수 (배치 처리용)
        프레임별 진폭 변화(std)를 계산하고, 미분 후 이동 평균 필터를 적용해 부드러운 움직임 특징을 생성.
        - 1단계: 프레임별 표준편차 계산
        - 2단계: 미분 + 절댓값 → 변화량 추출
        - 3단계: Overlap-save 이동 평균 필터 적용
    Parameters:
        detrended : Detrended amplitude 데이터 (frames x subcarriers)
        WIN_SIZE : 이동 평균 필터 크기 (기본값: 64)
    Example:
        detrended = detrending_amp(Hampel_filtered)
        feature = extract_motion_features(detrended, WIN_SIZE=64)
    """
    # 1단계: 진폭 변화량 계산 (프레임별 표준편차)
    std_per_pkt = np.std(detrended, axis=1)
    # 2단계: 변화량 미분 후 절댓값 처리
    feature_derivative_abs = np.abs(np.diff(std_per_pkt))
    # 3단계: Overlap-save convolution 기반 이동 평균 필터
    prev_samples = np.zeros(WIN_SIZE)
    padded_signal = np.concatenate([prev_samples, feature_derivative_abs])
    window = np.ones(WIN_SIZE)
    convolved = np.convolve(padded_signal, window, mode='valid')
    # 4단계: 최신 변화량만 반환
    feature = convolved[-len(feature_derivative_abs):]
    return feature

def detect_activity_with_ewma(feature, threshold_factor=2.5):
    """
    Desc:
        CSI 배치 데이터에서 EWMA 방식으로 임계선을 계산하고, 움직임 감지를 수행하는 함수 (배치 처리용)
        실시간 상태 저장 없이도 배치 단위에서 바로 실행 가능.
    Parameters:
        feature : 입력된 변화량 시계열 (1D array-like)
        threshold_factor : 평균 대비 임계 배율 (기본값: 2.5)
    Example:
        activity_flag, threshold = detect_activity_with_ewma(feature)
    """
    avgSigVal = np.mean(feature)
    ewma = avgSigVal  # 상태 유지 없이 한 번에 계산
    threshold = threshold_factor * ewma
    activity_flag = (feature > threshold).astype(float)
    return activity_flag, threshold

# =========================================================================================
# === 배치 처리를 위한 통합 파이프라인 ========================================================

def batch_cada_pipeline(amp_reduced, mu_bg, sigma_bg, use_filter_normalization=True, 
                       historical_window=100, WIN_SIZE=64, threshold_factor=2.5):
    """
    Desc:
        CADA 파이프라인을 배치 처리용으로 통합한 함수
        기존 main 함수의 처리 과정을 하나의 함수로 모듈화
    Parameters:
        amp_reduced : 채널이 제거된 진폭 데이터 (frames x subcarriers)
        mu_bg : 배경 평균값 (서브캐리어별)
        sigma_bg : 배경 표준편차 (서브캐리어별)
        use_filter_normalization : filter_normalization 사용 여부 (기본값: True)
        historical_window : 기준선 계산에 사용할 초기 프레임 수 (기본값: 100)
        WIN_SIZE : 이동 평균 필터 크기 (기본값: 64)
        threshold_factor : 평균 대비 임계 배율 (기본값: 2.5)
    Returns:
        results : 딕셔너리 형태의 결과
            - 'amp_normalized' : Z-score 정규화된 데이터
            - 'amp_filtered' : filter_normalization 적용된 데이터 (사용 시)
            - 'hampel_filtered' : Hampel 필터 적용된 데이터
            - 'detrended' : Detrending 적용된 데이터
            - 'feature' : 움직임 특징
            - 'activity_flag' : 활동 감지 플래그
            - 'threshold' : 임계값
    Example:
        # 캘리브레이션 데이터 로드
        mu_bg, sigma_bg = read_calibration(CALIB_PATH)
        
        # 파이프라인 실행
        results = batch_cada_pipeline(amp_reduced, mu_bg, sigma_bg)
        
        # 결과 사용
        feature = results['feature']
        activity_flag = results['activity_flag']
        threshold = results['threshold']
    """
    try:
        # 1. Z-score 정규화
        amp_normalized = z_normalization(amp_reduced, mu_bg, sigma_bg)
        
        # 2. Filter normalization (선택적)
        if use_filter_normalization:
            amp_filtered = filter_normalization(amp_normalized)
        else:
            amp_filtered = amp_normalized
        
        # 3. Hampel 필터 적용
        hampel_filtered = np.apply_along_axis(robust_hampel, 0, amp_filtered)
        
        # 4. Detrending
        detrended = detrending_amp(hampel_filtered, historical_window=historical_window)
        
        # 5. 움직임 특징 추출
        feature = extract_motion_features(detrended, WIN_SIZE=WIN_SIZE)
        
        # 6. 활동 감지
        activity_flag, threshold = detect_activity_with_ewma(feature, threshold_factor=threshold_factor)
        
        # 7. 결과 딕셔너리 생성
        results = {
            'amp_normalized': amp_normalized,
            'amp_filtered': amp_filtered,
            'hampel_filtered': hampel_filtered,
            'detrended': detrended,
            'feature': feature,
            'activity_flag': activity_flag,
            'threshold': threshold
        }
        
        return results
        
    except Exception as e:
        print(f"Error in batch_cada_pipeline: {e}")
        # 오류 발생 시 기본 결과 반환
        dummy_shape = amp_reduced.shape[0] - 1  # diff로 인해 길이가 1 줄어듦
        return {
            'amp_normalized': amp_reduced,
            'amp_filtered': amp_reduced,
            'hampel_filtered': amp_reduced,
            'detrended': amp_reduced,
            'feature': np.zeros(dummy_shape),
            'activity_flag': np.zeros(dummy_shape),
            'threshold': 0.1
        }

# ==========================================================================================
# === Payload parsing & Sliding-window CADA processor ======================================
# ==========================================================================================

def parse_and_normalize_payload(payload: str,
                                topic: str,
                                subcarriers: int,
                                indices_to_remove: list[int] | None,
                                mu_bg_dict: dict,
                                sigma_bg_dict: dict):
    """문자열 MQTT 페이로드에서 Z-score 정규화된 진폭 벡터와 타임스탬프를 추출한다.

    Returns (amp_z, packet_time) 또는 오류/데이터 부족 시 None.
    """
    try:
        # 1) 타임스탬프 파싱 ---------------------------------------------------
        match = re.search(r"time=(\d{15})", payload)
        if match:
            ts_str = match.group(1)
            packet_time = parse_custom_timestamp(ts_str)
        else:
            packet_time = datetime.now()

        # 2) CSI 문자열 → 복소수 배열 ------------------------------------------
        csi_data_str = payload.split("CSI values: ")[-1].strip()
        csi_values = list(map(int, csi_data_str.split()))
        if len(csi_values) < subcarriers * 2:
            return None  # 데이터 부족

        csi_complex = [csi_values[i] + 1j * csi_values[i + 1]
                       for i in range(0, len(csi_values), 2)]
        csi_complex = np.array(csi_complex)[:subcarriers]

        # 3) 노이즈 채널 제거 --------------------------------------------------
        if indices_to_remove:
            csi_complex = np.delete(csi_complex, indices_to_remove)

        csi_amplitude = np.abs(csi_complex)

        # 4) Z-score 정규화 ----------------------------------------------------
        if topic in mu_bg_dict and topic in sigma_bg_dict:
            amp_z = (csi_amplitude - mu_bg_dict[topic]) / sigma_bg_dict[topic]
        else:
            amp_z = csi_amplitude  # 캘리브 미존재 시 원본 유지

        return amp_z, packet_time

    except Exception as e:
        print(f"ERROR: parse_and_normalize_payload failed for {topic}: {e}")
        return None


class SlidingCadaCore:
    """320-frame 슬라이딩-윈도우 + stride 기반 CADA 배치 처리 헬퍼 클래스"""

    def __init__(self,
                 topic: str,
                 buffer_manager,
                 mu_bg_dict: dict,
                 sigma_bg_dict: dict,
                 window_size: int = 320,
                 stride: int = 40,
                 small_win_size: int = 64,
                 threshold_factor: float = 2.5,
                 executor: ThreadPoolExecutor | None = None):
        self.topic = topic
        self.buffer_manager = buffer_manager
        self.mu_bg_dict = mu_bg_dict
        self.sigma_bg_dict = sigma_bg_dict
        self.window_size = window_size
        self.stride = stride
        self.small_win_size = small_win_size  # WIN_SIZE for batch_cada_pipeline    
        self.threshold_factor = threshold_factor

        self._buf = deque(maxlen=self.window_size)
        self._ts_buf = deque(maxlen=self.window_size)
        self._counter = 0
        self._processing_running = False
        self._executor = executor or ThreadPoolExecutor(max_workers=1)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def push(self, amp_z: np.ndarray, packet_time):
        """프레임 하나를 버퍼에 추가하고 필요 시 비동기 배치 처리를 요청한다."""
        self._buf.append(amp_z.copy())
        self._ts_buf.append(packet_time)
        self._counter += 1

        if (len(self._buf) == self.window_size and
                (self._counter % self.stride == 0) and
                not self._processing_running):
            window_copy = np.array(self._buf)
            ts_copy = list(self._ts_buf)
            self._processing_running = True
            self._executor.submit(self._process_window, window_copy, ts_copy)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _process_window(self, csi_window: np.ndarray, ts_window):
        """백그라운드 스레드에서 실행: batch_cada_pipeline 후 buffer_manager에 결과 push"""
        try:
            if self.topic not in self.mu_bg_dict or self.topic not in self.sigma_bg_dict:
                print(f"WARNING: Calibration not found for {self.topic}. Skipping window.")
                return

            results = batch_cada_pipeline(
                amp_reduced=csi_window,
                mu_bg=self.mu_bg_dict[self.topic],
                sigma_bg=self.sigma_bg_dict[self.topic],
                use_filter_normalization=False,  # 이미 Z-score 완료
                historical_window=100,
                WIN_SIZE=self.small_win_size,
                threshold_factor=self.threshold_factor,
            )
            feature = results["feature"]  # 길이 = window_size - 1

            # ---- 원본 스크립트와 동일한 EWMA 기반 임계선 계산 ----
            avg_sig_val = float(np.mean(feature)) if len(feature) > 0 else 0.0
            alpha = 0.01
            prev_ewma = self.buffer_manager.cada_ewma_states.get(self.topic, 0.0)
            ewma_curr = avg_sig_val if prev_ewma == 0.0 else alpha * avg_sig_val + (1 - alpha) * prev_ewma
            self.buffer_manager.cada_ewma_states[self.topic] = ewma_curr
            Th = self.threshold_factor * ewma_curr

            activity_flag = (feature > Th).astype(float)

            # ---------------------------------------------------

            frames_to_push = min(self.stride, len(feature))
            start_idx = -frames_to_push
            for i in range(frames_to_push):
                idx = start_idx + i
                self.buffer_manager.cada_feature_buffers["activity_detection"][self.topic].append(feature[idx])
                self.buffer_manager.cada_feature_buffers["activity_flag"][self.topic].append(activity_flag[idx])
                self.buffer_manager.cada_feature_buffers["threshold"][self.topic].append(Th)

        except Exception as e:
            print(f"ERROR: SlidingCadaCore window processing failed for {self.topic}: {e}")
        finally:
            self._processing_running = False

# ======================== CADA 네임 최종 클래스 =========================
class SlidingCadaProcessor(SlidingCadaCore):  # type: ignore
    """SlidingCadaCore 를 외부 API 로 노출"""
    pass

# __all__ 업데이트
globals()["__all__"] = [name for name in globals().get("__all__", []) if "Atif" not in name]
if "SlidingCadaProcessor" not in globals()["__all__"]:
    globals()["__all__"].append("SlidingCadaProcessor")

if __name__ == "__main__" : 

    NO_ACTIVITY_CSI_PATH = r"data\raw\raw_noActivity_csi\merged_csi_data_noactivity.csv" 
    ACTIVITY_CSI_PATH = r"data\raw\raw_activity_csi\merged_csi_data_dooropen.csv" 
    CALIB_PATH = r"data\calibration\L0382_ESP_8_bg_params.csv" 

    FRAME_NUM = 500 # 6초
    SUBCARRIER_NUM = 52 # subcarrier 개수 (52로 고정)
    WIN_SIZE = 64
    TICK_SPACING = 10 # plot 시 tick 간격

    # 1. CSI 데이터 로드 및 전처리
    amp_activity, ts_activity = convert_csi_to_amplitude(ACTIVITY_CSI_PATH, SUBCARRIER_NUM)
    indices_to_remove = list(range(21,32))
    amp_reduced = np.delete(amp_activity, indices_to_remove, axis=1)
    
    # 2. 캘리브레이션 데이터 로드
    mu_bg, sigma_bg = read_calibration(CALIB_PATH)
    
    # 3. 배치 CADA 파이프라인 실행
    results = batch_cada_pipeline(
        amp_reduced=amp_reduced,
        mu_bg=mu_bg,
        sigma_bg=sigma_bg,
        use_filter_normalization=True,
        historical_window=100,
        WIN_SIZE=WIN_SIZE,
        threshold_factor=2.5
    )
    
    # 4. 결과 추출
    amp_normalized = results['amp_normalized']
    amp_filtered = results['amp_filtered']
    hampel_filtered = results['hampel_filtered']
    detrended = results['detrended']
    feature = results['feature']
    activity_flag = results['activity_flag']
    threshold = results['threshold']
    
    # 5. 시각화
    plot_csi_amplitude(amp_activity, ts_activity, "CSI Amplitude (Original)")
    plot_csi_amplitude(amp_reduced, ts_activity, "CSI Amplitude (Channel Reduced)")
    plot_csi_amplitude(amp_normalized, ts_activity, title="Z-Normalization")
    plot_csi_amplitude(amp_filtered, ts_activity, title="Z-Normalization ( Filtering )")
    plot_csi_amplitude(hampel_filtered, ts_activity, title="Hampel_filtered")
    plot_csi_amplitude(detrended, ts_activity, title="detrend_amp")
    plot_csi_amplitude(feature, ts_activity, title="Activity Detection with EWMA Threshold", amp2=activity_flag, amp3=threshold )