"""CADA_Live – 런처(Entry-point)
────────────────────────────────────────────
● 역할
    ▸ MQTT 수신, 실시간 분석, 플롯 출력까지
      *모든* 모듈을 한 줄씩 엮어 주는 실행 스크립트.

● 실행 흐름
    1. Queue 생성 → CADA_process.set_queue() 주입
    2. mqtt_utils.start_mqtt() 로 MQTT 스레드 가동
       ‣ 수신 콜백 = CADA_process.process_csi_data
    3. CADA_process.run_parallel_calibration()  (배경 캘리브레이션)
    4. run_plotter_universal()  – 메인 스레드에서 Plot 창 띄움
    5. Plot 창을 닫거나 Ctrl+C → _exit()  
       ‣ MQTT 루프 중지, Plot 닫기, 프로세스 종료(os._exit)

● 외부 설정
    BROKER_ADDRESS / BROKER_PORT / TOPICS    → 필요에 따라 수정

● 의존 모듈
    ▸ mqtt_utils.py     (브로커 연결)
    ▸ CADA_process.py   (실시간 분석)
    ▸ plot_utils.py     (그래프 갱신)
────────────────────────────────────────────
"""
from multiprocessing import Queue, freeze_support
import time
import signal
import sys
import mqtt_utils
import CADA_process as CADA
from plot_utils import run_plotter_universal

# ---- 설정 ----
BROKER_ADDRESS = "61.252.57.136"
BROKER_PORT    = 4991
TOPICS         = ["L0382/ESP/1","L0382/ESP/2","L0382/ESP/3","L0382/ESP/4","L0382/ESP/5","L0382/ESP/6","L0382/ESP/7","L0382/ESP/8" ]

def main():
    # ----  Plot용 Queue 만들고 알고리즘 쪽에 주입 ----
    result_queue = Queue()
    CADA.set_queue(result_queue)           # <-- 핵심 한 줄

    # ----  MQTT 시작 : 패킷 오면 CADA.process_csi_data 호출 ----
    mqtt_cli, mqtt_th = mqtt_utils.start_mqtt(
        TOPICS, BROKER_ADDRESS, BROKER_PORT, CADA.process_csi_data
    )
        
    # ----  캘리브레이션 & 메인 루프 ----
    CADA.run_parallel_calibration(False, 10*60*100)   # 필요시 조정

    # 4) 종료 처리 (Ctrl+C 또는 창 닫기)
    def _exit(sig=None, frm=None):
        try:
            mqtt_cli.loop_stop(force=True)   # paho-mqtt ≥1.6
        except TypeError:
            mqtt_cli.loop_stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _exit)
    signal.signal(signal.SIGTERM, _exit)

    # 5) 플롯 – 메인 스레드에서 실행 (창 닫히면 함수 리턴)
    run_plotter_universal(TOPICS, result_queue)
    _exit()          # 사용자가 창을 닫으면 여기로

if __name__ == "__main__":
    freeze_support()   # EXE 패키징 대비 (써도 무방)
    main()
