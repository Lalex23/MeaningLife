import cv2
import numpy as np
import pytesseract


class DisplayResultImage:
    def __init__(self, image):
        cv2.imshow('Image', image)

    def show(self):
        while True:
            if cv2.waitKey(0): # close window when a key press is detected
                break

    def __del__(self):
        cv2.destroyAllWindows()


class ExtractJavaCodeFromImage(object):
    PATH_TO_IMAGE = "java-code.png"

    def __init__(self):
        super().__init__()

        pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files\Tesseract-OCR\tesseract.exe'

    def get_code(self):
        prepared_image = self.prepare_image_erode_dilate()

        text = pytesseract.image_to_string(prepared_image, lang='eng', config='--psm 6 --oem 2')

        for string in text.split("\n"):
            print(string)

    def prepare_image_hsv_type(self):
        image = cv2.imread(self.PATH_TO_IMAGE)

        (H, W) = image.shape[:2]
        image_big = cv2.resize(image, (H*2, W*2))
        (HB, WB) = image_big.shape[:2]

        # (hMin = 0, sMin = 0, vMin = 81), (hMax = 179, sMax = 255, vMax = 255)

        hsv = cv2.cvtColor(image_big, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 91])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Create horizontal kernel and dilate to connect text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dilate = cv2.dilate(mask, kernel, iterations=5)

        image_result = 255 - cv2.bitwise_and(dilate, mask)

        return image_result

    def prepare_image_erode_dilate(self):
        # read image with openCv
        image_original = cv2.imread(self.PATH_TO_IMAGE)
        # Convert to GrayScale
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

        # Apply dilation and erosion to remove some noise
        kernel = np.ones((1, 1), np.uint8)
        image_dilate = cv2.dilate(image_gray, kernel, iterations=1)
        image_erode = cv2.erode(image_dilate, kernel, iterations=1)

        DisplayResultImage(image_dilate).show()
        DisplayResultImage(image_erode).show()

        image_result = cv2.bitwise_and(image_erode, image_erode)

        # Apply threshold to get image with only black and white
        #image_result = cv2.adaptiveThreshold(image_erode, 10, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return image_result
