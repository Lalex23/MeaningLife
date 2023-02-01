import cv2
import numpy as np
import pytesseract


class ExtractJavaCodeFromImage(object):
    def __init__(self):
        super().__init__()

        pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files\Tesseract-OCR\tesseract.exe'

    def get_code(self):
        image = cv2.imread("java-code.png")

        # Конвертация в серый
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # text = pytesseract.image_to_string(Image.open("java-code.png"), lang='eng',
                                           # config='--psm 1 --oem 1')

        (H, W) = image.shape[:2]
        image_big = cv2.resize(image, (H*2, W*2))
        # text = pytesseract.image_to_string(image_gray, lang='eng', config='--psm 6 --oem 1')
        image_blur = cv2.bilateralFilter(image_gray, 1, 125, 390)  # Blur to reduce noise


        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))

        hsv = cv2.cvtColor(image_big, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 218])
        upper = np.array([157, 54, 255])
        mask = cv2.inRange(hsv, lower, upper)
        dilate = cv2.dilate(mask, kernel, iterations=5)
        return_result, image_new = cv2.threshold(image_gray, 138, 355, cv2.THRESH_BINARY)

        # cv2.imshow('image_blur', image_blur)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # (hMin = 0, sMin = 0, vMin = 81), (hMax = 179, sMax = 255, vMax = 255)

        hsv = cv2.cvtColor(image_big, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 81])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Create horizontal kernel and dilate to connect text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dilate = cv2.dilate(mask, kernel, iterations=5)

        result = 255 - cv2.bitwise_and(dilate, mask)

        cv2.imshow('image_new', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        text = pytesseract.image_to_string(result, lang='eng', config='--psm 6 --oem 1')

        for string in text.split("\n"):
            print(string)

