# ── server_simple.py ───────────────────────────────────────────────────────
import autoroot
import os, json, asyncio, threading, time, queue
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from fastrtc import Stream
import cv2
from src.CADA.mqtt_utils import (
    start_mqtt_thread,
    DEFAULT_CSI_BROKER,
    DEFAULT_CSI_PORT,
    DEFAULT_CSI_TOPICS,
)

# ──────────────── 기본 파라미터 ────────────────
MQTT_HOST     = os.getenv("MQTT_BROKER", DEFAULT_CSI_BROKER)
MQTT_PORT     = int(os.getenv("MQTT_PORT", str(DEFAULT_CSI_PORT)))
# 여러 CSI 토픽 구독 (쉼표로 구분된 환경변수 사용 가능)
MQTT_TOPICS   = os.getenv("MQTT_TOPICS", ",".join(DEFAULT_CSI_TOPICS)).split(",")
SUBCARRIER_N  = 52
BUF_LEN       = 500                     # 최대 500 sample 버퍼만 유지
TOP_K         = 5                       # (지금은 의미 없지만 남겨둠)

# ──────────────── 전역 큐 (Thread :양방향_화살표: async) ────────────────
raw_q  = asyncio.Queue(maxsize=BUF_LEN)     # MQTT → Flask SocketIO

# ──────────────── 1. MQTT 메시지 핸들러 ────────────────
#  src/CADA/mqtt_utils.start_mqtt_thread 를 사용하여 백그라운드 수신.

def make_csi_handler(loop):
    """loop 변수 캡처하여 raw_q 로 CSI 시계열 push"""
    def _handler(_topic: str, payload: str):
        try:
            pkt = json.loads(payload)
            ts   = np.asarray(pkt.get("ts", []), dtype=np.float32) / 1e6  # μs → s
            csi  = np.asarray(pkt.get("csi", []), dtype=np.complex64).reshape(-1, SUBCARRIER_N)
            loop.call_soon_threadsafe(raw_q.put_nowait, (ts, csi))
        except Exception as e:
            print("[MQTT parse error]", e)

    return _handler

# ──────────────── 2. 아주 단순 전처리 coroutine ────────────────
async def csi_processing():
    """
    • 0 값이 전체인 sub-carrier 제거 
    • 절대값 → 모든 남은 채널 평균 → y(t) 
    • 그래프용 payload emit
    """
    while True:
        ts, csi = await raw_q.get()
        # 0-only sub-carrier drop (전부 0 이면 True)
        valid_mask = ~(csi == 0).all(axis=0)
        csi = csi[:, valid_mask]
        amp = np.abs(csi).mean(axis=1)      # (N,) 1-채널
        payload = {"t": ts.tolist(), "y": amp.tolist()}
        socketio.emit("csi", payload)        # Plotly 로 바로 송출
        await asyncio.sleep(0)               # 다른 task 에 컨텍스트 양보

# ──────────────── 3. Flask + SocketIO ────────────────
app      = Flask(__name__)
socketio = SocketIO(app, async_mode="eventlet")      # 기본 eventlet
@app.route("/")
def index(): return render_template("index_2.html")

# ──────────────── 4. FastRTC 영상 스트림 ────────────────
def fastrtc_thread():
    # FastRTC ≥0.1.5 에서는 Stream 객체를 직접 생성합니다.
    def cam_gen():
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

    stream = Stream(handler=cam_gen, modality="video", mode="receive")
    # FastRTC의 내장 Gradio UI로 스트림 노출 (별도 포트)
    stream.ui.launch(share=False, inbrowser=False)
    
# ──────────────── 5. 런처 ────────────────
def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # (1) MQTT Thread
    # MQTT 스레드 시작 (공용 유틸 사용)
    _, _cli = start_mqtt_thread(
        message_handler=make_csi_handler(loop),
        topics=MQTT_TOPICS,
        broker_address=MQTT_HOST,
        broker_port=MQTT_PORT,
        daemon=True,
    )
    # (2) FastRTC Thread
    threading.Thread(target=fastrtc_thread, daemon=True).start()
    # (3) Async processing task
    loop.create_task(csi_processing())
    socketio.run(app, host="0.0.0.0", port=5000)
if __name__ == "__main__":
    main()
# ───────────────────────────────────────────────────────────────────────────