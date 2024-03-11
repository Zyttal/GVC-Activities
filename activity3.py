import cv2
import numpy as np
import activity1
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image_used = "panda1.jpg"
    other_image = "panda3.jpeg"

    img1 = cv2.imread(image_used, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(other_image, cv2.IMREAD_UNCHANGED)

    # Original Image
    img1 = activity1.resizeImage(img1)
    img2 = activity1.resizeImage(img2)

    images = np.concatenate((img1, img2), axis=0)

    cv2_imshow(images)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Histogram of both images
    img1 = cv2.imread(image_used, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(other_image, cv2.IMREAD_GRAYSCALE)

    histogram1 = cv2.calcHist([img1],[0],None,[256],[0,256])
    histogram2 = cv2.calcHist([img2],[0],None,[256],[0,256])

    plt.hist(histogram1)

    plt.show()

    plt.hist(histogram2)

    plt.show()