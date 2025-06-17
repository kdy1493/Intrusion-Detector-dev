"""app_v1.py  ―  1세대 CADA + Flask + Socket.IO 실시간 대시보드
────────────────────────────────────────────────────────────
• CADA_process      : 실시간 CSI 분석 ➊
• mqtt_utils        : MQTT 수신 스레드 ➋
• Flask-SocketIO    : 분석 결과 웹으로 Push ➌
────────────────────────────────────────────────────────────
구조
    ┌── MQTT 브로커 ──► mqtt_utils.start_mqtt()
    │                    ↳ on_packet ⇒ CADA_process.process_csi_data
    │
    │     (multiprocessing.Queue)
    │                    ↘
    └── CADA_process  ──► result_queue.put(CSIResult …)
                          ↘
                _queue_drain_thread()
                          ↳ socketio.emit('cada_result', …)
"""

# ──────────────────────────────────────────────────────────
# ❶ 기본 세팅
# ──────────────────────────────────────────────────────────
import time, os, cv2, threading, queue
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
from multiprocessing import Queue

import CADA_process as CADA
import mqtt_utils


BROKER      = "61.252.57.136"
BROKER_PORT = 4991
TOPICS      = ["L0382/ESP/8"]          # 필요 시 늘리기
FPS_LIMIT   = 25                       # 웹 전송 최대 FPS

# ──────────────────────────────────────────────────────────
# ❷ Flask / SocketIO
# ──────────────────────────────────────────────────────────
app      = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")
result_q: Queue = Queue()          # CADA → 웹소켓 중계용
CADA.set_queue(result_q)

# ──────────────────────────────────────────────────────────
# ❸ MQTT 수신 스레드 시작
# ──────────────────────────────────────────────────────────
mqtt_cli, _ = mqtt_utils.start_mqtt(
    topics       = TOPICS,
    broker       = BROKER,
    port         = BROKER_PORT,
    on_packet    = CADA.process_csi_data
)

# ──────────────────────────────────────────────────────────
# ❹ 캘리브레이션 선-로드(필요 없다면 생략 가능)
# ──────────────────────────────────────────────────────────
CADA.run_parallel_calibration(False, 10*60*100)

# ──────────────────────────────────────────────────────────
# ❺ Queue → Socket.IO 브로드캐스트 스레드
# ──────────────────────────────────────────────────────────
def _queue_drain():
    last_emit = {}                  # topic → time.time()
    while True:
        try:
            item = result_q.get(timeout=0.5)
        except queue.Empty:
            continue

        now  = time.time()
        prev = last_emit.get(item.topic, 0.0)
        if (now - prev) < 1.0 / FPS_LIMIT:
            continue                # FPS 한도 넘어가면 skip
        last_emit[item.topic] = now

        socketio.emit(
            "cada_result",
            {
                "topic"       : item.topic,
                "timestamp_ms": int(item.timestamp.timestamp()*1000),
                "activity"    : float(item.activity_detection),
                "flag"        : int(item.activity_flag),
                "threshold"   : float(item.threshold),
            },
            namespace="/csi",
        )

threading.Thread(target=_queue_drain, daemon=True).start()

# ──────────────────────────────────────────────────────────
# ❻ Flask routes
# ──────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def gen():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Camera open failed")

        while True:
            ok, frame = cap.read()
            if not ok:  break
            cv2.putText(frame, time.strftime("%H:%M:%S"),
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,255), 2, cv2.LINE_AA)
            ret, buf = cv2.imencode(".jpg", frame)
            if not ret: continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        cap.release()

    return Response(gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame")


# ──────────────────────────────────────────────────────────
# ❼ Socket.IO(네임스페이스 /csi)
# ──────────────────────────────────────────────────────────
@socketio.on("connect", namespace="/csi")
def on_connect():
    print("[SocketIO] client connected")

@socketio.on("disconnect", namespace="/csi")
def on_disconnect():
    print("[SocketIO] client disconnected")

# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ctrl-C 시 MQTT 루프·프로세스 정리
    try:
        socketio.run(app, host="0.0.0.0", port=5001, debug=True)
    finally:
        try:
            mqtt_cli.loop_stop()
            mqtt_cli.disconnect()
        except Exception:
            pass
        os._exit(0)
