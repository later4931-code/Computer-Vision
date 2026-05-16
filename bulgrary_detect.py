import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

frames = []
gap = 5
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frames.append(gray)

    if len(frames) > gap + 1:
        frames.pop(0)

    cv.putText(frame, f"Frame count: {count}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if len(frames) > gap:
        diff = cv.absdiff(frames[0], frames[-1])
        _, thresh = cv.threshold(diff, 30, 255, cv.THRESH_BINARY)
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        motion = False
        for contour in contours:
            if cv.contourArea(contour) > 1000:
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                motion = True

        if motion:
            cv.putText(frame, "Motion Detected", (10, 70),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print("Motion Detected! Count:", count)
            count += 1

    cv.imshow("Motion Detection", frame)

    # Quit anytime with 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
