import cv2
import numpy as np
import math

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    if not ret:
        print("ERROR")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        area = cv2.contourArea(c)

        if area >= 1500:
            M = cv2.moments(c)

            if M["m00"] != 0:
                centrox = int(M["m10"] / M["m00"])
                centroy = int(M["m01"] / M["m00"])

                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int64(box)
                p1,p2,p3,p4 = box ##pode usar pra linha

                (cx, cy), (w, h), angle = rect
                if w>h:
                    angle += 90
                elif w<h:
                    angle+= 90
            
                cv2.drawContours(
                    frame,
                    [box],
                    0,
                    (0, 255, 255),
                    2
                )

                print(
                    "Blue found at: {},{} with area of: {} and angle of {}".format(
                        cx,
                        cy,
                        area,
                        angle
                    )
                )

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()