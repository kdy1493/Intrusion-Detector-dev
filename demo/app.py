import os
import time
import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
from sam2.build_sam import build_sam2_object_tracker
from alerts import AlertManager, AlertCodes

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

last_timestamp = "--:--:--"

class HumanDetectionApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.alert_manager = AlertManager()
        self.setup_config()
        self.setup_models()
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
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
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
        self.app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    app = HumanDetectionApp()
    app.run()