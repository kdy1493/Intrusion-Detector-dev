"""CADA_process — 실시간 CSI 처리 엔진
────────────────────────────────────────
● 기능
    1) MQTT payload  ➜  전처리(Z-score·Hampel)·피처추출  ➜  EWMA 임계값으로
       사람 움직임(0/1) 판단
    2) 판단 결과를 CSIResult 로 만들어 외부 multiprocessing.Queue 에 push

● 공개 API
    set_queue(q)                         : Plot 프로세스가 소비할 Queue 주입
    run_parallel_calibration(force, n)   : 배경 μ·σ CSV 저장/로드
    process_csi_data(topic, payload)     : MQTT on_message 에서 직접 호출

● 버퍼·슬라이딩 윈도우
    WIN_SIZE                =  64        # 이동평균 커널  
    ADD_BUFFER_SIZE         = 128        # Queue 전송 단위  
    PROCESSING_BUFFER_SIZE  = 320        # 128*2 + 64  

    ▶ 패킷이 도착할 때마다 topic 별 deque 에 append  
    ▶ 길이가 320 프레임이 되면  
        • **최근 320 프레임**으로 특징·임계값 계산  
        • **가장 최근 128 프레임**(ADD_BUFFER_SIZE) 결과를 Queue 로 push  
        • deque.popleft() ×128 → 버퍼엔 192 프레임이 남음  
    ▶ 이후 128 프레임이 추가되어 다시 320 이 되면 동일 과정 반복

● 처리 파이프라인
    ① CSI 진폭 추출 & 중앙 11 서브캐리어 제거  
    ② μ·σ(Z-score) 정규화          (센서별 캘리브레이션 값 사용)  
    ③ Robust Hampel Filter         (서브캐리어별 스파이크 완화)  
    ④ 프레임별 표준편차 → 1차 차분 → |·| → 64-tap 이동평균  
    ⑤ EWMA(α = 0.01)로 동적 Baseline, Th = 2.5 × EWMA  
       변화량 > Th  → activity_flag = 1  else 0

● CSIResult 필드
    timestamp, activity_detection(float), activity_flag(0/1),
    threshold(float), topic(str)

● 다중 토픽 지원
    모든 상태 구조가 topic key 로 분리되어 있어
    TOPICS 리스트만 늘리면 추가 센서를 그대로 처리.
────────────────────────────────────────
"""

import re   # 정규표현식 모듈
import os
import csv
import time
import numpy as np
from datetime import datetime
from collections import deque
from scipy.signal import medfilt

TOPICS = ["L0382/ESP/1","L0382/ESP/2","L0382/ESP/3","L0382/ESP/4","L0382/ESP/5","L0382/ESP/6","L0382/ESP/7","L0382/ESP/8" ]

# --- Calibration Parameters ---
FORCE_NEW_CALIBRATION = False  # Set to False to use existing calibration
CALIBRATION_SAMPLES = 10*60*100 
CALIB_DIR = r"data\CSI_Calibration"
os.makedirs(CALIB_DIR, exist_ok=True)

# --- Process Parameters ---
SUBCARRIERS = 52
WIN_SIZE = 64
ADD_BUFFER_SIZE =  40              
PROCESSING_BUFFER_SIZE = ADD_BUFFER_SIZE* 2 + WIN_SIZE     # 320

# --- Queue 생성 ---
result_queue = None
def set_queue(q):      #  ← 런처에서 호출
    global result_queue
    result_queue = q

# --- CSIResult class ---
class CSIResult:
    def __init__(self, timestamp, activity_detection, activity_flag, threshold, topic):
        self.timestamp = timestamp
        self.activity_detection = activity_detection
        self.activity_flag = activity_flag
        self.threshold = threshold
        self.topic = topic

# --- Processing buffer ---
processing_buffer = {topic: deque(maxlen=PROCESSING_BUFFER_SIZE) for topic in TOPICS}   # 양쪽에서 뺄 수 있는 큐 구조의 임시 버퍼 ( 길이 초과 시 : 오래된 데이터 삭제 )
packet_timestamps = {topic: deque(maxlen=PROCESSING_BUFFER_SIZE) for topic in TOPICS}   
mean_buffer = {topic: deque(maxlen=100) for topic in TOPICS}
prev_samples = {topic: np.zeros(WIN_SIZE) for topic in TOPICS}

# --- EMA parameters ---
ewma_avg_dict = {topic: 0.0 for topic in TOPICS}


# --- Calibration parameters ---
mu_bg_dict = {}
sigma_bg_dict = {}

# --- Calibration functions ---
def calibrate_background(topic, calibration_data, force_new=False):
    '''배경 데이터 보정
    평균, 표준편차 계산'''
    calib_file = os.path.join(CALIB_DIR, f"{topic.replace('/','_')}_bg_params.csv")
    if not force_new and os.path.exists(calib_file):        
        with open(calib_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        mu_bg = np.array([float(x) for x in rows[0]])          
        sigma_bg = np.array([float(x) for x in rows[1]])       
        print(f"[{topic}] Loaded: {calib_file}")
    else:                                                   
        mu_bg = np.mean(calibration_data, axis=0)           
        sigma_bg = np.std(calibration_data, axis=0)         
        sigma_bg[sigma_bg == 0] = 1                         
        with open(calib_file, 'w', newline='') as f:   
            writer = csv.writer(f)
            writer.writerow(mu_bg)
            writer.writerow(sigma_bg)
        print(f"[{topic}] Saved: {calib_file}")
    return mu_bg, sigma_bg

def run_parallel_calibration(FORCE_NEW_CALIBRATION, CALIBRATION_SAMPLES):
    '''병렬 캘리브레이션'''
    global mu_bg_dict, sigma_bg_dict

    if FORCE_NEW_CALIBRATION:                                                                               
        calibration_buffers = {topic: [] for topic in TOPICS}                                               
        calibration_done = {topic: False for topic in TOPICS}                                               

        print("Collecting No Movement Data...")                                                             
        while not all(calibration_done.values()):                                                           
            for topic in TOPICS :
                if not calibration_done[topic]:                                                             
                    if len(processing_buffer[topic]) > 0:                                                   
                        calibration_buffers[topic].append(processing_buffer[topic][-1])                     
                    if len(calibration_buffers[topic]) >= CALIBRATION_SAMPLES:                              
                        calibration_done[topic] = True                                                      
                        print(f"Collected {len(calibration_buffers[topic])} samples for topic {topic}.")    

            time.sleep(0.01)  

        for topic in TOPICS:                                                                                 
            calibration_data = np.array(calibration_buffers[topic])                                         
            mu_bg, sigma_bg = calibrate_background(topic, calibration_data, force_new=True)                 
            mu_bg_dict[topic] = mu_bg                                                                      
            sigma_bg_dict[topic] = sigma_bg

        print("Calibration complete for all topics. Now running real-time processing...")

    else:                                                                                                 
        for topic in TOPICS:
            dummy = np.zeros((1, SUBCARRIERS - len(range(21, 32))))
            mu_bg, sigma_bg = calibrate_background(topic, dummy, force_new=False)                         
            mu_bg_dict[topic] = mu_bg                                                                      
            sigma_bg_dict[topic] = sigma_bg
            print(f"Loaded calibration for {topic}...")

        print("Calibration loaded. Starting real-time processing...")

# --- Timestamp parsing ---
def parse_custom_timestamp(topic, payload):
    '''timestamp 파싱 : ESP timestamp -> datetime'''
    match = re.search(r'time=(\d{15})', payload)    
    if match:
        ts_str = match.group(1)
        year = 2000 + int(ts_str[0:2])
        month = int(ts_str[2:4])
        day = int(ts_str[4:6])
        hour = int(ts_str[6:8])
        minute = int(ts_str[8:10])
        second = int(ts_str[10:12])
        millisecond = int(ts_str[12:15])
        microsecond = millisecond * 1000
        packet_time = datetime(year, month, day, hour, minute, second, microsecond)
        return packet_time
    else:
        packet_time = datetime.now()
        return packet_time
    
# --- Parse CSI payload ---
def parse_csi_payload(payload):
    '''CSI 데이터 파싱 및 전처리'''
    # CSI 데이터 파싱
    csi_data_str = payload.split("CSI values: ")[1].strip()    
    csi_values = list(map(int, csi_data_str.split()))          
    if len(csi_values) < SUBCARRIERS * 2:                      
        return None

    # 복소수 변환 및 진폭 계산
    csi_complex = [csi_values[i] + 1j * csi_values[i + 1] for i in range(0, len(csi_values), 2)]  
    csi_amplitude = np.array([np.abs(x) for x in csi_complex])[:SUBCARRIERS]                      
    
    # 중앙 서브캐리어 제거
    indices_to_remove = list(range(21, 32))                                                       
    csi_amplitude_reduced = np.delete(csi_amplitude, indices_to_remove)                          

    return csi_amplitude_reduced

# --- Normalize and buffer ---
def normalize_and_buffer(topic, packet_time, csi_amplitude_reduced):
    '''Z-score 정규화 및 버퍼 관리'''
    global processing_buffer, packet_timestamps
    # Z-score 정규화
    if topic in mu_bg_dict and topic in sigma_bg_dict:                                          
        z_normalized = (csi_amplitude_reduced - mu_bg_dict[topic]) / sigma_bg_dict[topic]       
    else:
        z_normalized = csi_amplitude_reduced                                                     

    # 버퍼에 추가
    processing_buffer[topic].append(z_normalized.copy())                                         

    return len(processing_buffer[topic]) == PROCESSING_BUFFER_SIZE

# --- Robust Hampel Filter ---
def robust_hampel_filter(column, window_size=5, n_sigma=3):
    '''이상치 제거 필터'''
    median = medfilt(column, kernel_size=window_size)       
    deviation = np.abs(column - median)                     
    mad = np.median(deviation)                              
    threshold = n_sigma * mad                               
    outliers = deviation > threshold                        
    column[outliers] = median[outliers]                     
    return column

# --- Process activity detection ---
def process_activity_detection(topic, proc_array):
    '''활동 감지 처리'''
    # Hampel 필터 적용
    amp_filtered = np.apply_along_axis(robust_hampel_filter, 0, proc_array)                  
    mean_per_packet = np.mean(amp_filtered, axis=1, keepdims=True)                           
    detrended_packet = amp_filtered - mean_per_packet                                       

    # 기준선 계산
    mean_current = np.mean(amp_filtered, axis=0)                                             
    if len(mean_buffer[topic]) > 0:                                                          
        hist_array = np.array(mean_buffer[topic])                                           
        mean_historical = np.mean(hist_array, axis=0)                                          
    else:                                                                                    
        mean_historical = mean_current                                                          
    combined_mean = (mean_current + mean_historical) / 2                                     

    # 변화량 계산
    detrended = detrended_packet - combined_mean                                             
    SCAbsEuclidSumFeatured = np.std(detrended, axis=1)
    FeaturedDerivative = np.diff(SCAbsEuclidSumFeatured)
    FeaturedDerivativeAbs = np.abs(FeaturedDerivative)

    # 이동평균 필터링
    padded_signal = np.concatenate([prev_samples[topic], FeaturedDerivativeAbs])            
    window = np.ones(WIN_SIZE)                                                              
    convolved = np.convolve(padded_signal, window, mode='valid')                            
    prev_samples[topic] = FeaturedDerivativeAbs[-WIN_SIZE:]                                 
    
    FeaturedDerivativeAbsSum = convolved[-len(FeaturedDerivativeAbs):]                      

    return FeaturedDerivativeAbsSum

# --- Calculate threshold and detect ---
def calculate_threshold_and_detect(topic, FeaturedDerivativeAbsSum):
    '''임계값 계산 및 활동 감지'''
    # EMA 계산
    avgSigVal = np.mean(FeaturedDerivativeAbsSum) if len(FeaturedDerivativeAbsSum) > 0 else 0            
    alpha = 0.01                                                                        
    if ewma_avg_dict[topic] == 0.0:
        ewma_avg_dict[topic] = avgSigVal
    else:
        ewma_avg_dict[topic] = alpha * avgSigVal + (1 - alpha) * ewma_avg_dict[topic]   
    
    # 임계값 계산 및 활동 감지
    Th = 2.5 * ewma_avg_dict[topic]                                     
    ActivityDetected = (FeaturedDerivativeAbsSum > Th).astype(float)    

    return FeaturedDerivativeAbsSum, ActivityDetected, Th

# --- Process CSI data ---
def process_csi_data(topic, payload):
    '''CSI 데이터 처리 메인 함수'''
    global packet_timestamps
    try:
        # 1. timestamp 파싱
        packet_time = parse_custom_timestamp(topic, payload)
        packet_timestamps[topic].append(packet_time)

        # 2. CSI 데이터 파싱 및 전처리
        csi_amplitude_reduced = parse_csi_payload(payload)

        # 3. 데이터 정규화 및 버퍼 관리
        buffer_full = normalize_and_buffer(topic, packet_time, csi_amplitude_reduced)
        if not buffer_full:
            return

        # 4. 활동 감지 처리
        proc_array = np.array(processing_buffer[topic])
        timestamps_array = list(packet_timestamps[topic])
        FeaturedDerivativeAbsSum = process_activity_detection(topic, proc_array)

        # 5. 임계값 계산 및 활동 감지
        FeaturedDerivativeAbsSum, ActivityDetected, Th = calculate_threshold_and_detect(topic, FeaturedDerivativeAbsSum)

        # 6. 프레임별로 Queue에 전송
        for i in range(ADD_BUFFER_SIZE):                 # 0~127 전송
            result_queue.put(
                CSIResult(
                    timestamp=timestamps_array[i],   # np.diff ⇒ +1
                    activity_detection=float(FeaturedDerivativeAbsSum[i]),
                    activity_flag=float(ActivityDetected[i]),
                    threshold=float(Th),
                    topic=topic
                )
            )

        # 이미 처리한 프레임을 버퍼에서 제거해 중복 방지
        for _ in range(ADD_BUFFER_SIZE):
            processing_buffer[topic].popleft()
            packet_timestamps[topic].popleft()

    except Exception as e:
        print(f"Error processing CSI data for topic {topic}: {e}")

