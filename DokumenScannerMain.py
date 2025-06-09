import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image

# ===== Fungsi Stack Gambar =====
def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0]) if isinstance(imgArray[0], list) else 1
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1] if rowsAvailable else imgArray[0].shape[1]
    height = imgArray[0][0].shape[0] if rowsAvailable else imgArray[0].shape[0]
    width = int(width * scale)
    height = int(height * scale)

    if rowsAvailable:
        for x in range(rows):
            for y in range(cols):
                img = imgArray[x][y]
                img = cv2.resize(img, (width, height))
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                imgArray[x][y] = img
        hor = [np.hstack(imgArray[x]) for x in range(rows)]
        ver = np.vstack(hor)
    else:
        for x in range(rows):
            img = imgArray[x]
            img = cv2.resize(img, (width, height))
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            imgArray[x] = img
        ver = np.hstack(imgArray)

    if len(labels) != 0 and rowsAvailable:
        eachImgWidth = width
        eachImgHeight = height
        for d in range(rows):
            for c in range(cols):
                label = str(labels[d][c])
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(label) * 13 + 27, 30 + eachImgHeight * d),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, label, (c * eachImgWidth + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2)
    return ver

# ===== Fungsi Kontur Terbesar =====
def getBiggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area

# ===== Fungsi Reorder Titik =====
def reorder(points):
    points = points.reshape((4, 2))
    newPoints = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[2] = points[np.argmax(diff)]
    return newPoints

# ===== Fungsi Simpan =====
def saveImagesAndPDF(images_dict):
    folder = "scans"
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jpg_paths = []

    for name, image in images_dict.items():
        path = os.path.join(folder, f"{name}_{timestamp}.jpg")
        cv2.imwrite(path, image)
        jpg_paths.append(path)

    pil_images = [Image.open(p).convert("RGB") for p in jpg_paths]
    pdf_path = os.path.join(folder, f"Scan_{timestamp}.pdf")
    pil_images[0].save(pdf_path, save_all=True, append_images=pil_images[1:])
    print(f"✅ Disimpan sebagai JPG dan PDF: {pdf_path}")

# ===== Main Program =====
webCamFeed = False
pathImage = "1.jpg"  # Ubah sesuai gambar
cap = cv2.VideoCapture(0)
cap.set(10, 160)
heightImg = 640
widthImg = 480

while True:
    if webCamFeed:
        success, img = cap.read()
        if not success:
            break
    else:
        img = cv2.imread(pathImage)

    img = cv2.resize(img, (widthImg, heightImg))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 150)

    kernel = np.ones((5, 5))
    imgDilated = cv2.dilate(imgCanny, kernel, iterations=2)
    imgEroded = cv2.erode(imgDilated, kernel, iterations=1)

    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, _ = cv2.findContours(imgEroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 2)

    biggest, _ = getBiggestContour(contours)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 10)

        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarp = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        imgWarpGray = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY)
        imgAdaptive = cv2.adaptiveThreshold(imgWarpGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
        imgAdaptive = cv2.bitwise_not(imgAdaptive)

        imgHistEq = cv2.equalizeHist(imgWarpGray)

        kernel_sharp = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])
        imgSharpened = cv2.filter2D(imgWarpGray, -1, kernel_sharp)
    else:
        imgBigContour = np.zeros_like(img)
        imgWarp = np.zeros_like(img)
        imgWarpGray = np.zeros_like(imgGray)
        imgAdaptive = np.zeros_like(imgGray)
        imgHistEq = np.zeros_like(imgGray)
        imgSharpened = np.zeros_like(imgGray)

    imageArray = [
        [img, imgGray, imgCanny, imgDilated],
        [imgEroded, imgContours, imgBigContour, imgWarp],
        [imgWarpGray, imgAdaptive, imgHistEq, imgSharpened]
    ]
    labels = [
        ["Original", "Gray", "Canny", "Dilated"],
        ["Eroded", "Contours", "Biggest", "Warp"],
        ["Warp Gray", "Adaptive", "HistogramEq", "Sharpened"]
    ]

    stacked = stackImages(imageArray, 0.4, labels)
    cv2.imshow("Document Scanner", stacked)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        saveImagesAndPDF({
            "Original": img,
            "Gray": imgGray,
            "Canny": imgCanny,
            "Dilated": imgDilated,
            "Eroded": imgEroded,
            "Contours": imgContours,
            "BiggestContour": imgBigContour,
            "WarpPerspective": imgWarp,
            "WarpGray": imgWarpGray,
            "AdaptiveThreshold": imgAdaptive,
            "HistogramEqualization": imgHistEq,
            "Sharpened": imgSharpened
        })

cap.release()
cv2.destroyAllWindows()
