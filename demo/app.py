import autoroot
import os
import time
import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from sam2.build_sam import build_sam2_object_tracker
from alerts import AlertManager, AlertCodes
import math
from flask_socketio import SocketIO
from src.CADA.realtime_csi_handler_utils import create_buffer_manager, load_calibration_data
from src.CADA.CADA_process import parse_and_normalize_payload, SlidingCadaProcessor
from src.CADA.mqtt_utils import start_csi_mqtt_thread

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------------------------------------------------
# SocketIO & CADA 글로벌 초기화
# ------------------------------------------------------------------

socketio = SocketIO(async_mode="threading")  # app 객체는 뒤에서 bind
_mqtt_started = False  # MQTT 스레드 시작 여부 플래그

# --- CADA 설정값 ---
TOPICS = ["L0382/ESP/8"]
INDICES_TO_REMOVE = list(range(21, 32))  # 52→41
SUBCARRIERS = 52
CADA_WINDOW_SIZE = 320
CADA_STRIDE = 40
SMALL_WIN_SIZE = 64
FPS_LIMIT = 10
BROKER_ADDR = "61.252.57.136"
BROKER_PORT = 4991

# CADA 전역 버퍼 및 프로세서 (app 생성 이후에 초기화할 예정)
buf_mgr = None  # type: ignore
sliding_processors = {}
time_last_emit = {}
th_interp_state = {}

# timestamp global for /timestamp route
last_timestamp = "--:--:--"

class HumanDetectionApp:
    def __init__(self):
        self.app = Flask(__name__)
        socketio.init_app(self.app, async_mode="threading")
        self.alert_manager = AlertManager()
        self.setup_config()
        self.setup_models()
        # ---- CADA 버퍼 초기화 ----
        global buf_mgr, sliding_processors
        buf_mgr = create_buffer_manager(TOPICS)
        load_calibration_data(TOPICS, buf_mgr.mu_bg_dict, buf_mgr.sigma_bg_dict)
        for t in TOPICS:
            buf_mgr.cada_ewma_states[t] = 0.0

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
            ) for t in TOPICS
        }
        self.setup_routes()
        self.reset_state()

    def setup_config(self):
        self.YOLO_MODEL_PATH = os.path.abspath("checkpoints/yolov8n.pt")
        self.SAM_CONFIG_PATH = "./configs/samurai/sam2.1_hiera_b+.yaml"
        self.SAM_CHECKPOINT_PATH = os.path.abspath("checkpoints/sam2.1_hiera_base_plus.pt")
        self.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.HFOV = 67.6 
        self.VFOV = 41.2

    def setup_models(self):
        self.yolo = YOLO(self.YOLO_MODEL_PATH)
        self.sam = None

    def reset_state(self): 
        self.detection_mode = True
        self.was_tracking = False
        self.stationary_timer_start = None
        self.last_center = None

    def setup_routes(self):
        self.app.route('/')(self.index)
        self.app.route('/video_feed')(self.video_feed)
        self.app.route('/alerts')(self.alerts)
        self.app.route('/redetect', methods=['POST'])(self.redetect)
        self.app.route('/timestamp')(self.timestamp)

    def timestamp(self):
        return jsonify({'timestamp': last_timestamp})
    
    def index(self): 
        return render_template('index.html')

    def alerts(self):
        def event_stream():
            self.alert_manager.send_alert(AlertCodes.SYSTEM_STARTED, "SYSTEM_STARTED: waiting for human")
            while True:
                data = self.alert_manager.get_next_alert()
                if data:
                    yield f"data: {data}\n\n"
                else:
                    yield "data: \n\n"
        return Response(event_stream(), mimetype='text/event-stream')

    def redetect(self):
        self.reset_state()
        return jsonify({'success': True})

    def process_detection(self, frame):
        persons = []
        results = self.yolo.predict(frame, classes=[0], device=self.DEVICE, verbose=False)
        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append([[x1, y1], [x2, y2]])

        if persons:
            self.sam = build_sam2_object_tracker(
                num_objects=len(persons),
                config_file=self.SAM_CONFIG_PATH,
                ckpt_path=self.SAM_CHECKPOINT_PATH,
                device=self.DEVICE,
                verbose=False
            )
            self.sam.track_new_object(
                img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                box=np.array(persons)
            )
            self.was_tracking = True
            self.detection_mode = False
            self.alert_manager.send_alert(AlertCodes.PERSON_DETECTED, "PERSON_DETECTED")
            return True
        return False

    def process_masks(self, m_np, disp, frame):
        h, w = disp.shape[:2]
        bbox_coords = None
        for i in range(m_np.shape[0]):
            mask = (m_np[i,0] > 0.5).astype(np.uint8)
            if mask.sum() == 0:
                continue
            mask = cv2.resize(mask, (w, h), cv2.INTER_NEAREST) 
            disp[mask>0] = (0,255,0)
            ys, xs = np.where(mask>0)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            bbox_coords = (x1, y1, x2, y2)
            cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,255), 2)
        disp = cv2.addWeighted(disp, 0.3, frame, 0.7, 0)
        return bbox_coords

    def check_stationary_behavior(self, bbox_coords):
        if bbox_coords is None:
            return
            
        x1, y1, x2, y2 = bbox_coords
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        now = time.time()

        if self.last_center is None:
            self.last_center = (cx, cy)
            self.stationary_timer_start = now
        else:
            dist = np.hypot(cx - self.last_center[0], cy - self.last_center[1])
            if dist < 5:
                if self.stationary_timer_start and now - self.stationary_timer_start >= 3.0:
                    self.alert_manager.send_alert(AlertCodes.STATIONARY_BEHAVIOR, "STATIONARY BEHAVIOR DETECTED: analysis required")
                    self.stationary_timer_start = None
            else:
                self.last_center = (cx, cy)
                self.stationary_timer_start = now

    def process_tracking(self, frame, disp):
        out = self.sam.track_all_objects(img=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        masks = out.get("pred_masks")
        has_mask = False

        if masks is not None:
            m_np = masks.cpu().numpy()
            for i in range(m_np.shape[0]):
                if (m_np[i,0] > 0.5).sum() > 0:
                    has_mask = True
                    break

            if has_mask:
                bbox_coords = self.process_masks(m_np, disp, frame)

                x1, y1, x2, y2 = bbox_coords
                cx, cy = (x1+x2)/2, (y1+y2)/2

                cv2.circle(disp, (int(cx), int(cy)), radius=5, color=(0,0,255), thickness=-1)

                h, w = disp.shape[:2]
                dx = cx - (w/2)
                dy = cy - (h/2)
                pan_angle  = (dx/(w/2)) * (self.HFOV/2)
                tilt_angle = (dy/(h/2)) * (self.VFOV/2)
                print(f"Pan: {pan_angle:.1f}°, Tilt: {tilt_angle:.1f}°")

                self.check_stationary_behavior(bbox_coords)

        if self.was_tracking and not has_mask:
            self.alert_manager.send_alert(AlertCodes.PERSON_LOST, "PERSON_LOST")
            self.reset_state()

        return has_mask

    def gen_frames(self):
        global last_timestamp
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError("camera is not opened")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            h, w = frame.shape[:2]
            now = time.time()
            ts_str = time.strftime("%H:%M:%S", time.localtime(now))
            last_timestamp = ts_str  

            text_size, _ = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_w, text_h = text_size

            disp = frame.copy()
            cv2.putText(
                disp,
                ts_str,
                (w - text_w - 10, text_h + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            if self.detection_mode: 
                detected = self.process_detection(frame)
                if detected:
                    for r in self.yolo.predict(frame, classes=[0], device=self.DEVICE, verbose=False):
                        for b in r.boxes:
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)
            elif self.sam is not None:
                self.process_tracking(frame, disp)

            ret2, buf = cv2.imencode('.jpg', disp)
            if not ret2:
                continue
            frame_bytes = buf.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes +
                   b'\r\n')
            time.sleep(0.01)

        cap.release()

    def video_feed(self):
        return Response(self.gen_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    def run(self):
        socketio.run(self.app, host="0.0.0.0", port=5000, debug=True)

# ------------------------------------------------------------------
# MQTT + SocketIO global helpers
# ------------------------------------------------------------------

def _start_mqtt():
    """백그라운드에서 MQTT 수신 쓰레드를 한 번만 시작"""
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
        payload, topic, SUBCARRIERS, INDICES_TO_REMOVE,
        buf_mgr.mu_bg_dict, buf_mgr.sigma_bg_dict)
    if parsed is None:
        return
    amp_z, pkt_time = parsed
    buf_mgr.timestamp_buffer[topic].append(pkt_time)
    sliding_processors[topic].push(amp_z, pkt_time)

    if not buf_mgr.cada_feature_buffers["activity_detection"][topic]:
        return

    idx = -1
    activity = buf_mgr.cada_feature_buffers["activity_detection"][topic][idx]
    flag     = buf_mgr.cada_feature_buffers["activity_flag"][topic][idx]
    threshold = buf_mgr.cada_feature_buffers["threshold"][topic][idx]
    ts_ms = int(pkt_time.timestamp()*1000)

    if (now - prev_emit) < 1.0/FPS_LIMIT:
        return
    time_last_emit[topic] = now

    socketio.emit("cada_result", {
        "topic": topic,
        "timestamp_ms": ts_ms,
        "activity": float(activity),
        "flag": int(flag),
        "threshold": float(threshold),
    }, namespace="/csi")

# ------------------------------------------------------------------
# SocketIO 네임스페이스 이벤트
# ------------------------------------------------------------------

@socketio.on("connect", namespace="/csi")
def on_connect():
    _start_mqtt()
    print("[SocketIO] Client connected")

@socketio.on("disconnect", namespace="/csi")
def on_disconnect():
    print("[SocketIO] Client disconnected")

# ------------------------------------------------------------------
if __name__ == '__main__':
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    HumanDetectionApp().run()