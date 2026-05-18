import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque

model = YOLO('yolov8n.pt')
cap = cv.VideoCapture('yyuy.mp4')

# Resize video dimensions
frame_width = 800
frame_height = 600

id_map = {}
nex_id = 1

trail = defaultdict(lambda: deque(maxlen=30))
appear = defaultdict(int)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize frame
    frame = cv.resize(frame, (frame_width, frame_height))
    
    res = model.track(frame, classes = [0], persist=True, verbose=False)
    annotated_frame = frame.copy() ##So .copy() lets you draw annotations without destroying the original frame data.

    if res[0].boxes is not None:  ### .boxes is for bounding boxes
        boxes = res[0].boxes.xyxy.numpy()
        ids = res[0].boxes.id.numpy() 
        
        for box, id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            appear[id] += 1
            if appear[id] > 5 and id not in id_map:
                id_map[id] = nex_id
                nex_id += 1

            if id in id_map:
                sid = id_map[id]
                trail[id].append((cx, cy))
                cv.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(annotated_frame, f'ID: {sid}', (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                
                # Draw trail line
                if len(trail[id]) > 1:
                    pts = np.array(list(trail[id]), dtype=np.int32)
                    cv.polylines(annotated_frame, [pts], False, (255, 0, 0), 2)

    cv.imshow('Annotated Frame', annotated_frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
