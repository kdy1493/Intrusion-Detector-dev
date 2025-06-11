#!/usr/bin/env python3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
from sam2.build_sam import build_sam2_object_tracker

YOLO_MODEL_PATH     = os.path.abspath("checkpoints/yolov8n.pt")
SAM_CONFIG_PATH     = "./configs/samurai/sam2.1_hiera_t.yaml"
SAM_CHECKPOINT_PATH = os.path.abspath("checkpoints/sam2.1_hiera_tiny.pt")
DEVICE              = "cuda:0" if torch.cuda.is_available() else "cpu"
CAM_INDEX           = 2

MAX_PEOPLE          = 10
REDETECT_INTERVAL_S = 2.0
IOU_THRESHOLD       = 0.3

def draw_bboxes(img, boxes, color=(0,255,255)):
    for (x1,y1),(x2,y2) in boxes:
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

def compute_iou(boxA, boxB):
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])
    interW = max(0, xB-xA)
    interH = max(0, yB-yA)
    interArea = interW * interH
    areaA = (boxA[1][0]-boxA[0][0])*(boxA[1][1]-boxA[0][1])
    areaB = (boxB[1][0]-boxB[0][0])*(boxB[1][1]-boxB[0][1])
    union = areaA + areaB - interArea
    return interArea/union if union>0 else 0

def main():
    yolo = YOLO(YOLO_MODEL_PATH)
    cap  = cv2.VideoCapture(CAM_INDEX,
           cv2.CAP_DSHOW if os.name=="nt" else cv2.CAP_V4L2)
    if not cap.isOpened():
        print("camera detection failed"); return
    cv2.namedWindow("HumanSeg", cv2.WINDOW_NORMAL)

    initial_boxes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release(); return

        results = yolo.predict(frame, classes=[0], device=DEVICE, verbose=False)
        for res in results:
            for b in res.boxes:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                initial_boxes.append([[x1,y1],[x2,y2]])
        if initial_boxes:
            print(f"person detected")
            break

        cv2.putText(frame, "waiting for human", (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imshow("HumanSeg", frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            cap.release(); cv2.destroyAllWindows(); return

    sam = build_sam2_object_tracker(
        num_objects=MAX_PEOPLE,
        config_file=SAM_CONFIG_PATH,
        ckpt_path=SAM_CHECKPOINT_PATH,
        device=DEVICE,
        verbose=False
    )
    first_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sam.track_new_object(img=first_rgb, box=np.array(initial_boxes))

    tracked_boxes = initial_boxes.copy()
    last_redetect = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        out   = sam.track_all_objects(img=img_rgb)
        masks = out.get('pred_masks')
        h,w   = frame.shape[:2]
        overlay = frame.copy()
        curr_boxes = []
        if masks is not None:
            m_np = masks.cpu().numpy()
            for i in range(m_np.shape[0]):
                mask = (m_np[i,0]>0.5).astype(np.uint8)
                if mask.max()==0: continue
                mask = cv2.resize(mask,(w,h),cv2.INTER_NEAREST)
                overlay[mask>0] = (0,255,0)
                ys,xs = np.where(mask>0)
                curr_boxes.append([[xs.min(),ys.min()],[xs.max(),ys.max()]])

        now = time.time()
        if now - last_redetect >= REDETECT_INTERVAL_S:
            last_redetect = now
            results = yolo.predict(frame, classes=[0], device=DEVICE, verbose=False)
            new_boxes = []
            for res in results:
                for b in res.boxes:
                    box = [[int(b.xyxy[0][0]),int(b.xyxy[0][1])],
                           [int(b.xyxy[0][2]),int(b.xyxy[0][3])]]
                    new_boxes.append(box)

            for box in new_boxes:
                if not tracked_boxes or \
                   max(compute_iou(box, tb) for tb in tracked_boxes) < IOU_THRESHOLD:
                    sam.track_new_object(img=img_rgb, box=np.array([box]))
                    tracked_boxes.append(box)
                    print("person detected", box)

        draw_bboxes(overlay, curr_boxes)
        disp = cv2.addWeighted(overlay,0.5,frame,0.5,0)
        cv2.imshow("HumanSeg", disp)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
