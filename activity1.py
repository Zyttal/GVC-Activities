import cv2
import numpy as np
import random

def printImageProperties(img):
    print(f"Shape of Image (Rows, Columns & Channels): {img.shape}")
    print(f"Total Number of Pixels: {img.size}")
    print(f"Data type: {img.dtype}")

    color_value = img[300, 300] # x = 300, y = 300 for fixed pixel

    print(f"Color Value of Image: {color_value}\n")

def resizeImage(img):
    dim = (640, 480)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resized_img

if __name__ == "__main__":

    image_used = "panda1.jpg"
    bnw_image = "bw_image.png"

    # Open Image and Display Properties

    img = cv2.imread(image_used, cv2.IMREAD_COLOR)
    img = resizeImage(img)

    cv2.imshow("Colored Image", img)

    print("COLORED IMAGE PROPERTIES")
    printImageProperties(img)

    cv2.waitKey(0)

    # Convert to Grayscale with IMREAD_GRAYSCALE flag

    img_gray = cv2.imread(image_used, cv2.IMREAD_GRAYSCALE)
    img_gray = resizeImage(img_gray)

    cv2.imshow("Grayscaled Image", img_gray)

    print("GRAYSCALED IMAGE PROPERTIES")
    printImageProperties(img_gray)

    # Convert Grayscaled Image to black and white (Binary)

    (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite('bw_image.png', img_bw)

    cv2.waitKey(0)

    # Read and show the created Black and white image

    img = cv2.imread(bnw_image, cv2.IMREAD_UNCHANGED)
    img = resizeImage(img)

    cv2.imshow("Black and White Image", img)

    print("BLACK AND WHITE IMAGE PROPERTIES")
    printImageProperties(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



