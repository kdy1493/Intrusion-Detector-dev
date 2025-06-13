"""
CADA_visualizer.py
----
Wi-Fi CSI 실시간 시각화 메인 스크립트.

주요 기능
----------
• MQTT 패킷 수신 및 처리
• CADA 알고리즘 기반 실시간 활동 탐지
• 실시간 데이터 시각화

실행 방법
----------
```bash
python scripts/CADA_visualizer.py
```
"""

import autorootcwd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import numpy as np

from src.CADA.mqtt_utils import start_csi_mqtt_thread
from src.CADA.plot_utils import plot_realtime_universal
from src.CADA.buffer_manager_utils import create_buffer_manager
from src.CADA.CADA_process import (
    parse_csi_data,
    z_normalization,
    cada_pipeline,
    run_parallel_calibration
)

# ==================== 설정 ====================
BROKER_ADDRESS = "61.252.57.136"
BROKER_PORT = 4991
TOPICS = ["L0382/ESP/8"]
SUBCARRIERS = 52
INDICES_TO_REMOVE = list(range(21, 32))
WINDOW_SIZE = 64
BUFFER_SIZE = 4096
CADA_WINDOW_SIZE = 320
CADA_STRIDE = 40
THRESHOLD_FACTOR = 2.5
ALPHA = 0.01
# ==============================================

# 전역 객체/상태    
buffer_manager = None
calibration_done = False
executor = ThreadPoolExecutor(max_workers=len(TOPICS))
sliding_windows = {topic: deque(maxlen=CADA_WINDOW_SIZE) for topic in TOPICS}
frame_counters = {topic: 0 for topic in TOPICS}
mu_bg_dict = None
sigma_bg_dict = None

def process_mqtt_data(topic: str, payload: str):
    global mu_bg_dict, sigma_bg_dict
    if not calibration_done or buffer_manager is None:
        return
    # 1. CSI 데이터 파싱
    result = parse_csi_data(payload, SUBCARRIERS, INDICES_TO_REMOVE)
    if result is None:
        return
    csi_amplitude, timestamp = result

    # 2. Z-score 정규화
    z_normalized = z_normalization(csi_amplitude, topic, mu_bg_dict, sigma_bg_dict)
    sliding_windows[topic].append(z_normalized)
    frame_counters[topic] += 1

    # 3. stride마다 CADA 알고리즘 실행
    if len(sliding_windows[topic]) == CADA_WINDOW_SIZE and frame_counters[topic] % CADA_STRIDE == 0:
        amp_window = np.stack(sliding_windows[topic], axis=0)  # (윈도우크기, subcarriers)
        # CADA 처리
        features, activity_flags, thresholds, mean_buffer, prev_samples, ewma_state = cada_pipeline(
            amp_window,
            topic=topic,
            mean_buffer=buffer_manager.mean_buffers[topic],
            prev_samples=buffer_manager.prev_samples[topic],
            ewma_state=buffer_manager.ewma_states[topic],
            historical_window=100,
            win_size=WINDOW_SIZE,
            threshold_factor=THRESHOLD_FACTOR,
            alpha=ALPHA
        )
        # stride 구간의 각 프레임별 값을 버퍼에 저장
        for i in range(len(features)):
            buffer_manager.add_csi_data(topic, z_normalized, timestamp)  # 최신 프레임만 저장(혹은 필요시 amp_window[i], timestamp 등으로 수정)
            buffer_manager.add_feature(topic, 'activity_detection', features[i])
            buffer_manager.add_feature(topic, 'activity_flag', activity_flags[i])
            buffer_manager.add_feature(topic, 'threshold', thresholds[i])

def update_plot(frame):
    if not calibration_done or buffer_manager is None:
        plot_realtime_universal(None, None, None, waiting_message="System initializing...")
        return
    try:
        plot_realtime_universal(
            feature_buffers=buffer_manager.get_features(),
            timestamp_buffer=buffer_manager.timestamp_buffer,
            topics=TOPICS,
            plot_points=BUFFER_SIZE,
        )
    except Exception as e:
        print(f"Plot error: {e}")
        plot_realtime_universal(None, None, None, waiting_message=f"Plot error: {str(e)}")

if __name__ == "__main__":
    print("Starting WiFi CSI CADA Activity Visualizer...")
    print("Initializing data buffers...")
    buffer_manager = create_buffer_manager(TOPICS, BUFFER_SIZE, WINDOW_SIZE)
    print("Loading calibration data...")
    mu_bg_dict, sigma_bg_dict = run_parallel_calibration(TOPICS)
    calibration_done = True
    print("System initialization completed!")
    print("Starting MQTT connection...")
    mqtt_thread, client = start_csi_mqtt_thread(
        message_handler=process_mqtt_data,
        topics=TOPICS,
        broker_address=BROKER_ADDRESS,
        broker_port=BROKER_PORT,
    )
    print("Waiting for data reception...")
    print("Starting realtime visualization...")
    fig = plt.figure(figsize=(16, 10))
    ani = FuncAnimation(fig, update_plot, interval=1000, cache_frame_data=False)
    plt.suptitle("WiFi CSI CADA Activity Visualizer", fontsize=20)
    plt.show() 