"""Точка входа"""
from utils import ExtractJavaCodeFromImage


def main():
    extractor = ExtractJavaCodeFromImage()
    extractor.get_code()


if __name__ == '__main__':
    main()
