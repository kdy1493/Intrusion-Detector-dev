# ── server_simple.py ───────────────────────────────────────────────────────
import os, json, asyncio, threading, time, queue
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from fastrtc import Stream              # 웹캠 Stream
# ──────────────── 기본 파라미터 ────────────────
MQTT_HOST     = os.getenv("MQTT_BROKER", "localhost")
MQTT_TOPIC    = "csi/raw"               # 브로커에서 받는 토픽
SUBCARRIER_N  = 52
BUF_LEN       = 500                     # 최대 500 sample 버퍼만 유지
TOP_K         = 5                       # (지금은 의미 없지만 남겨둠)
# ──────────────── 전역 큐 (Thread :양방향_화살표: async) ────────────────
raw_q  = asyncio.Queue(maxsize=BUF_LEN)     # MQTT → Flask SocketIO

# ──────────────── 1. MQTT 리시버 Thread ────────────────
def mqtt_thread(loop):
    def on_connect(cli, _, __, rc): cli.subscribe(MQTT_TOPIC)
    def on_message(cli, _, msg):
        try:
            pkt = json.loads(msg.payload.decode())
            # pkt = { "ts": [...],  "csi": [...] }  --- (ts μs, csi ( N*52 ))
            ts   = np.asarray(pkt["ts"], dtype=np.float32) / 1e6  # s
            csi  = np.asarray(pkt["csi"], dtype=np.complex64).reshape(-1, SUBCARRIER_N)
            loop.call_soon_threadsafe(raw_q.put_nowait, (ts, csi))
        except Exception as e:
            print(":x: MQTT parse error:", e)
    cli = mqtt.Client()
    cli.on_connect, cli.on_message = on_connect, on_message
    cli.connect(MQTT_HOST, 1883, 60)
    cli.loop_forever()

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
def index(): return render_template("index.html")

# ──────────────── 4. FastRTC 영상 스트림 ────────────────
def fastrtc_thread():
    stream = Stream.video(width=640, height=480, fps=30)
    stream.mount(socketio.server)            # SocketIO 서버에 붙이기
    stream.ui.launch(block=True, inbrowser=False)

# ──────────────── 5. 런처 ────────────────
def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # (1) MQTT Thread
    threading.Thread(target=mqtt_thread, args=(loop,), daemon=True).start()
    # (2) FastRTC Thread
    threading.Thread(target=fastrtc_thread, daemon=True).start()
    # (3) Async processing task
    loop.create_task(csi_processing())
    socketio.run(app, host="0.0.0.0", port=5000)
if __name__ == "__main__":
    main()
# ───────────────────────────────────────────────────────────────────────────