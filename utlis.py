import cv2
import numpy as np

def initializeTrackbars():
    pass  # Jika tidak digunakan, bisa dikosongkan

def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    return imgCanny

def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest = np.array([])
    maxArea = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    return biggest

def getWarp(img, points):
    width, height = 480, 640
    pts1 = np.float32(points.reshape(4, 2))
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))
    return imgOutput

def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)

    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                img = imgArray[x][y]
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                imgArray[x][y] = cv2.resize(img, (0, 0), None, scale, scale)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)

        if len(labels) != 0:
            eachImgWidth = int(ver.shape[1] / cols)
            eachImgHeight = int(ver.shape[0] / rows)
            for d in range(rows):
                for c in range(cols):
                    label = str(labels[d][c])
                    cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                                  (c * eachImgWidth + len(label) * 13 + 27, 30 + eachImgHeight * d),
                                  (255, 255, 255), cv2.FILLED)
                    cv2.putText(ver, label, (c * eachImgWidth + 10, eachImgHeight * d + 20),
                                cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
        return ver
    else:
        raise Exception("stackImages expects a 2D list (list of list of images)")
