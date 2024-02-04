from email.mime import image
import time
import os
import cv2
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


folderPath = "Z:\KunalSir\Finger_Detection\FingerImages"
myList = os.listdir(folderPath)

overlayList = []
for imagePath in myList:
    fullPath = f"{folderPath}/{imagePath}"
    image = cv2.imread(fullPath)
    overlayList.append(image)
    print(f"{folderPath}/{imagePath}")

pTime = 0

detector = htm.handDetector()
while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    tipIds = [4, 8, 12, 16, 20]
    if len(lmList) != 0:
        fingers = []
        # thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(fingers)

        totalfingers = fingers.count(1)
        print(totalfingers)
        h, w, c = overlayList[totalfingers-1].shape
        img[0:h, 0:w] = overlayList[totalfingers]

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS{int(fps)}', (400, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
