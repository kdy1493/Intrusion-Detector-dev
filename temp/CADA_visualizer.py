"""
CADA_visualizer.py
----
Wi-Fi CSI 실시간 시각화 메인 스크립트.

주요 기능
----------
• MQTT 패킷 버퍼링 기능.
• SlidingCadaProcessor 기반 실시간 활동 탐지 기능.
• Matplotlib FuncAnimation 기반 1초 주기 그래프 갱신 기능.
• 외부 스트리머 접근 지원 기능.
• 실행 방법 안내 기능.

실행 방법
----------
가상환경을 활성화한 후 다음 명령으로 실행합니다.

```bash
python scripts/realtime_visualizer_3.py
```

브로커 주소·포트, 토픽, 버퍼 크기 등은 파일 상단의 **설정 블록**을 수정하여 적용할 수 있음. 
"""

import autorootcwd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from concurrent.futures import ThreadPoolExecutor

from src.CADA.mqtt_utils import start_csi_mqtt_thread
from src.CADA.plot_utils import plot_realtime_universal
from src.CADA.realtime_csi_handler_utils import (
    create_buffer_manager, load_calibration_data,
)

# ==================== 설정 ====================
BROKER_ADDRESS = "61.252.57.136"
BROKER_PORT = 4991
TOPICS = ["L0382/ESP/8"]
SUBCARRIERS = 52
INDICES_TO_REMOVE = list(range(21, 32))
WINDOW_SIZE = 64
BUFFER_SIZE = 4096

# CADA 슬라이딩-윈도우 파라미터
CADA_WINDOW_SIZE = 320        # 기존 배치 크기 그대로 유지
CADA_STRIDE = 40              # 40프레임마다 계산 → 지연 ≤ 40프레임

CSV_WRITE_FRAME_COUNT = 128
# ==============================================

# 전역 객체/상태
buffer_manager = None

# SlidingCadaProcessor 초기화용
from src.CADA.CADA_process import parse_and_normalize_payload, SlidingCadaProcessor

# 공용 ThreadPoolExecutor (토픽 수만큼)
executor = ThreadPoolExecutor(max_workers=len(TOPICS))
sliding_processors = {}

# Calibration flag 초기화
calibration_done = False

# -------------------------------------------------
# Helper: MQTT 수신 콜백
# -------------------------------------------------

def process_mqtt_data(topic: str, payload: str):
    """실시간 CADA 슬라이딩 윈도우 처리 전용"""
    if not calibration_done or buffer_manager is None:
        return

    parsed = parse_and_normalize_payload(
        payload=payload,
        topic=topic,
        subcarriers=SUBCARRIERS,
        indices_to_remove=INDICES_TO_REMOVE,
        mu_bg_dict=buffer_manager.mu_bg_dict,
        sigma_bg_dict=buffer_manager.sigma_bg_dict,
    )
    if parsed is None:
        return

    z_normalized, packet_time = parsed
    # 타임스탬프 버퍼에 저장 (그래프 X축용)
    buffer_manager.timestamp_buffer[topic].append(packet_time)
    # SlidingCadaProcessor에 push → 내부적으로 stride 체크 및 배치 처리
    sliding_processors[topic].push(z_normalized, packet_time)

# -------------------------------------------------
# Plot 업데이트 루프
# -------------------------------------------------

def update_plot(frame):
    if not calibration_done or buffer_manager is None:
        plot_realtime_universal(None, None, None, waiting_message="System initializing...")
        return

    try:
        combined_features = buffer_manager.get_combined_features()
        plot_realtime_universal(
            feature_buffers=combined_features,
            timestamp_buffer=buffer_manager.timestamp_buffer,
            topics=TOPICS,
            plot_points=4096,
        )
    except Exception as e:
        print(f"Plot error: {e}")
        import traceback

        traceback.print_exc()
        plot_realtime_universal(None, None, None, waiting_message=f"Plot error: {str(e)}")

# -------------------------------------------------
# 메인 엔트리 포인트
# -------------------------------------------------

if __name__ == "__main__":
    print("Starting realtime+low-latency fusion WiFi CSI visualization system (ver.3)...")
    enabled_methods = ["CADA Activity Detection (sliding)"]
    if not enabled_methods:
        print("ERROR: At least one method must be enabled!")
        exit(1)
    print(f"Enabled methods: {', '.join(enabled_methods)}")
    print(
        f"Mode: CADA Activity Detection (sliding window size={CADA_WINDOW_SIZE}, stride={CADA_STRIDE})"
    )
    print(f"Settings: buffer_size={BUFFER_SIZE}, window_size={WINDOW_SIZE}")

    # 버퍼 매니저 초기화 & 캘리브레이션 로드
    print("Initializing data buffers...")
    buffer_manager = create_buffer_manager(TOPICS, BUFFER_SIZE, WINDOW_SIZE)

    print("Loading calibration data...")
    load_calibration_data(TOPICS, buffer_manager.mu_bg_dict, buffer_manager.sigma_bg_dict)
    for topic in TOPICS:
        buffer_manager.cada_ewma_states[topic] = 0.0
    print("CADA EWMA states reset to 0.0 for clean start")

    # SlidingCadaProcessor 인스턴스 생성 (토픽별)
    sliding_processors = {
        topic: SlidingCadaProcessor(
            topic=topic,
            buffer_manager=buffer_manager,
            mu_bg_dict=buffer_manager.mu_bg_dict,
            sigma_bg_dict=buffer_manager.sigma_bg_dict,
            window_size=CADA_WINDOW_SIZE,
            stride=CADA_STRIDE,
            small_win_size=WINDOW_SIZE,
            threshold_factor=2.5,
            executor=executor,
        )
        for topic in TOPICS
    }

    calibration_done = True
    print("System initialization completed!")

    # MQTT 연결 시작
    print("Starting MQTT connection...")
    mqtt_thread, client = start_csi_mqtt_thread(
        message_handler=process_mqtt_data,
        topics=TOPICS,
        broker_address=BROKER_ADDRESS,
        broker_port=BROKER_PORT,
    )
    print("Waiting for data reception...")
    print("Starting realtime visualization (Matplotlib UI)…")

    fig = plt.figure(figsize=(16, 10))
    ani = FuncAnimation(fig, update_plot, interval=1000, cache_frame_data=False)
    title = "WiFi CSI CADA Activity Visualizer (ver.3) - Low-latency Sliding Window"
    plt.suptitle(title, fontsize=20)
    plt.show() 