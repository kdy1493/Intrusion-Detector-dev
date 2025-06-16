import autoroot
import os
import time
from flask import Flask, render_template, Response, jsonify
from alerts import AlertManager, AlertCodes
from flask_socketio import SocketIO
import cv2
from temp.realtime_csi_handler_utils import (
    create_buffer_manager,
    load_calibration_data,
)
from temp.CADA_process import (
    parse_and_normalize_payload,
    SlidingCadaProcessor,
)
from temp.mqtt_utils import start_csi_mqtt_thread

# ------------------------------------------------------------------
# CADA 설정값 -------------------------------------------------------
# ------------------------------------------------------------------
TOPICS = ["L0382/ESP/8"]
INDICES_TO_REMOVE = list(range(21, 32))  # 52→41
SUBCARRIERS = 52
CADA_WINDOW_SIZE = 320
CADA_STRIDE = 40
SMALL_WIN_SIZE = 64
FPS_LIMIT = 10
BROKER_ADDR = "61.252.57.136"
BROKER_PORT = 4991

socketio = SocketIO(async_mode="threading")  # Flask 객체는 클래스에서 바인딩
_mqtt_started = False  # MQTT 스레드 시작 여부 플래그

# 전역 버퍼/프로세서 -------------------------------------------------
buf_mgr = None  # type: ignore
sliding_processors = {}
time_last_emit = {}

# ------------------------------------------------------------------
# Flask 앱 + CADA 실시간 시각화 전용 클래스 ---------------------------
# ------------------------------------------------------------------
class CadaWebApp:
    def __init__(self):
        self.app = Flask(__name__)
        socketio.init_app(self.app, async_mode="threading")
        self.alert_manager = AlertManager()
        self._setup_cada()
        self._setup_routes()
    # --------------------------------------------------------------
    # CADA 초기화 ---------------------------------------------------
    # --------------------------------------------------------------
    def _setup_cada(self):
        global buf_mgr, sliding_processors
        buf_mgr = create_buffer_manager(TOPICS)
        load_calibration_data(TOPICS, buf_mgr.mu_bg_dict, buf_mgr.sigma_bg_dict)
        for t in TOPICS:
            buf_mgr.cada_ewma_states[t] = 0.0  # clean start

        sliding_processors = {
            t: SlidingCadaProcessor(
                topic=t,
                buffer_manager=buf_mgr,
                mu_bg_dict=buf_mgr.mu_bg_dict,
                sigma_bg_dict=buf_mgr.sigma_bg_dict,
                window_size=CADA_WINDOW_SIZE,
                stride=CADA_STRIDE,
                small_win_size=SMALL_WIN_SIZE,
                threshold_factor=2.5,
            )
            for t in TOPICS
        }
    # --------------------------------------------------------------
    # Flask Routes -------------------------------------------------
    # --------------------------------------------------------------
    def _setup_routes(self):
        self.app.route("/")(self.index)
        self.app.route("/alerts")(self.alerts)
        self.app.route("/timestamp")(self.timestamp)
        self.app.route("/video_feed")(self.video_feed)
        self.app.route("/redetect", methods=["POST"])(self.redetect)

    def index(self):
        # 단순 Plotly+SocketIO 대시보드 템플릿 사용
        return render_template("index.html")

    def alerts(self):
        """Server-Sent Events 로 간단 알림 전송"""

        def event_stream():
            # 시스템 시작 알림 1회 송출
            self.alert_manager.send_alert(
                AlertCodes.SYSTEM_STARTED, "SYSTEM_STARTED"
            )
            while True:
                data = self.alert_manager.get_next_alert()
                yield f"data: {data}\n\n" if data else "data: \n\n"

        return Response(event_stream(), mimetype="text/event-stream")

    def timestamp(self):
        return jsonify({"timestamp": time.strftime("%H:%M:%S")})

    # --------------------------------------------------------------
    # Video stream -------------------------------------------------
    # --------------------------------------------------------------
    def gen_frames(self):
        """웹캠 프레임을 Motion JPEG 스트림으로 전송"""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("Unable to open camera (index 0)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Timestamp overlay (HH:MM:SS)
            ts_text = time.strftime("%H:%M:%S")
            cv2.putText(
                frame,
                ts_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            ret2, buf = cv2.imencode(".jpg", frame)
            if not ret2:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )

        cap.release()

    def video_feed(self):
        return Response(
            self.gen_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def redetect(self):
        """프론트엔드 호환용 더미 엔드포인트(언제든 확장 가능)"""
        return jsonify({"success": True})

    # --------------------------------------------------------------
    def run(self):
        socketio.run(self.app, host="0.0.0.0", port=5000, debug=True)


# ------------------------------------------------------------------
# MQTT + SocketIO helpers (전역) ------------------------------------
# ------------------------------------------------------------------

def _start_mqtt():
    """백그라운드 MQTT 수신 스레드 한 번만 시작"""

    global _mqtt_started
    if _mqtt_started:
        return

    start_csi_mqtt_thread(
        message_handler=mqtt_handler,
        topics=TOPICS,
        broker_address=BROKER_ADDR,
        broker_port=BROKER_PORT,
        daemon=True,
    )
    _mqtt_started = True


def mqtt_handler(topic: str, payload: str):
    now = time.time()
    prev_emit = time_last_emit.get(topic, 0.0)

    parsed = parse_and_normalize_payload(
        payload,
        topic,
        SUBCARRIERS,
        INDICES_TO_REMOVE,
        buf_mgr.mu_bg_dict,
        buf_mgr.sigma_bg_dict,
    )
    if parsed is None:
        return

    amp_z, pkt_time = parsed
    buf_mgr.timestamp_buffer[topic].append(pkt_time)
    sliding_processors[topic].push(amp_z, pkt_time)

    # 실시간 CADA 특징이 없다면 아직 초기화 중이므로 return
    if not buf_mgr.cada_feature_buffers["activity_detection"][topic]:
        return

    idx = -1  # latest
    activity = buf_mgr.cada_feature_buffers["activity_detection"][topic][idx]
    flag = buf_mgr.cada_feature_buffers["activity_flag"][topic][idx]
    threshold = buf_mgr.cada_feature_buffers["threshold"][topic][idx]
    ts_ms = int(pkt_time.timestamp() * 1000)

    # FPS 제한
    if (now - prev_emit) < 1.0 / FPS_LIMIT:
        return
    time_last_emit[topic] = now

    socketio.emit(
        "cada_result",
        {
            "topic": topic,
            "timestamp_ms": ts_ms,
            "activity": float(activity),
            "flag": int(flag),
            "threshold": float(threshold),
        },
        namespace="/csi",
    )


# ------------------------------------------------------------------
# SocketIO 네임스페이스 이벤트 --------------------------------------
# ------------------------------------------------------------------


@socketio.on("connect", namespace="/csi")
def on_connect():
    _start_mqtt()
    print("[SocketIO] Client connected")


@socketio.on("disconnect", namespace="/csi")
def on_disconnect():
    print("[SocketIO] Client disconnected")


# ------------------------------------------------------------------
# main --------------------------------------------------------------
# ------------------------------------------------------------------
if __name__ == "__main__":
    CadaWebApp().run()