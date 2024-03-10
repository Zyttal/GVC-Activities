import cv2
import numpy as np

def printImageProperties(img):
    print(f"Shape of Image (Rows, Columns & Channels): {img.shape}")
    print(f"Total Number of Pixels: {img.size}")
    print(f"Data type: {img.dtype}\n")

if __name__ == "__main__":

    image_used = "panda1.jpg"
    bnw_image = "bw_image.png"

    # Open Image and Display Properties

    img = cv2.imread(image_used, cv2.IMREAD_COLOR)
    dim = (1024, 768)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Colored Image", resized_img)

    print("COLORED IMAGE PROPERTIES")
    printImageProperties(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert to Grayscale with IMREAD_GRAYSCALE flag

    img_gray = cv2.imread(image_used, cv2.IMREAD_GRAYSCALE)
    dim = (1024, 768)
    resized_img_gray = cv2.resize(img_gray, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Grayscaled Image", resized_img_gray)

    print("GRAYSCALED IMAGE PROPERTIES")
    printImageProperties(img_gray)

    # Convert Grayscaled Image to black and white (Binary)

    (thresh, img_bw) = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite('bw_image.png', img_bw)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Read and show the created Black and white image

    img = cv2.imread(bnw_image, cv2.IMREAD_UNCHANGED)
    dim = (1024, 768)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Black and White Image", resized_img)

    print("BLACK AND WHITE IMAGE PROPERTIES")
    printImageProperties(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



