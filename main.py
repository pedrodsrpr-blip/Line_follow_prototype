import cv2
import numpy as np

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

    img = cv2.bitwise_and(frame, frame, mask=mask)

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    #cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    for c in contours:
        area = cv2.contourArea(c)

        if area >= 1500:
            M = cv2.moments(c)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int64(box)

                # (coordenadas), w, h, angle = box
                # Se função, poderia desempacotar e retornar aqui

                corners = np.int32(c)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                corners = cv2.goodFeaturesToTrack(
                    gray,
                    100,
                    1.0,
                    0.01,
                    mask
                )
                
                #for i in corners:
                   # x, y = i.ravel()

                    #cv2.line(
                     #   frame,
                      #  x,
                    #    y,
                   #     (255, 0, 0),
                   #     thickness=1,
                   #     lineType=cv2.LINE_8,
                   #     shift=0
                   # )
                
                cv2.drawContours(
                    frame,
                    [box],
                    0,
                    (0, 0, 255),
                    2
                )

                print(
                    "Blue found at: {},{} with area of: {}".format(
                        cx,
                        cy,
                        area
                    )
                )

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()