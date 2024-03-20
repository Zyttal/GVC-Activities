import cv2
import numpy as np

def resizeImage(img):
    dim = (640, 480)
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized_img

def main():
    image_used = "panda1.jpg"

    img = cv2.imread(image_used)
    img = resizeImage(img)

    for i in range(img.shape[0]): 
        for j in range(img.shape[1]): 
            if i % 10 == 0 and j % 10 == 0:
                img[i,j] = 0 

    cv2.imshow('Modified Pixels (Image)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()
