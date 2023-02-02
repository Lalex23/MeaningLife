import cv2
import numpy as np
import pytesseract


class DisplayResultImage:
    def __init__(self, image):
        cv2.imshow('Image', image)

    @staticmethod
    def show():
        while True:
            if cv2.waitKey(0):  # close window when a key press is detected
                break

    def __del__(self):
        cv2.destroyAllWindows()


class TextAnalysis:
    def __init__(self, text):
        self.text = text
        self.code = ""
        self.file_name = None

    def make(self):
        for count, line in enumerate(self.text.split('\n'), start=1):
            new_line = " ".join(line.split(' ')[1::])
            print(repr(new_line))
            if new_line.startswith('import'):
                self.code += new_line + "\n\r"
            elif new_line.startswith(' public class') and new_line.endswith('{'):
                self.code += "\r" + new_line.strip() + "\n\t"
            elif new_line.startswith('private final'):
                self.code += "\t" + new_line.strip() + "\n"
            if count == 10:
                break

        print(self.code)

    def get_code(self):
        return self.code


class ExtractJavaCodeFromImage(object):
    PATH_TO_IMAGE = "java-code.png"

    def __init__(self):
        super().__init__()
        # Укажи путь до исполняемого файла tesseract
        pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files\Tesseract-OCR\tesseract.exe'

    def get_code(self):
        prepared_image = self.prepare_image_filter_a()

        DisplayResultImage(prepared_image).show()

        custom_configure = '-l eng --psm 6 --oem 2'
        text = pytesseract.image_to_string(prepared_image, config=custom_configure)

        # for string in text.split("\n"):
        #     print(string)

        text_analysis = TextAnalysis(text)
        text_analysis.make()
        code = text_analysis.get_code()

        #print(code)

    def prepare_image_filter_b(self):
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

    def prepare_image_filter_a(self):
        # read image with openCv
        image_original = cv2.imread(self.PATH_TO_IMAGE)

        # Convert to GrayScale
        image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

        # Apply dilation and erosion to remove some noise
        kernel = np.ones((15, 15), np.uint8)
        image_dilate = cv2.dilate(image_gray, kernel, iterations=1)
        image_erode = cv2.erode(image_dilate, kernel, iterations=1)

        image_result = 250 - cv2.bitwise_and(image_gray, image_gray, mask=image_erode)

        return image_result

