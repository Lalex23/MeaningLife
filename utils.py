import cv2
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

        # image_big = cv2.resize(image_gray, (35800, 26500))
        # text = pytesseract.image_to_string(image_gray, lang='eng', config='--psm 6 --oem 1')
        image_blur = cv2.bilateralFilter(image_gray, 2, 50, 250)  # Blur to reduce noise
        cv2.imshow('image', image_blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        text = pytesseract.image_to_string(image_blur, lang='eng', config='--psm 6 --oem 1')

        for string in text.split("\n"):
            print(string)

