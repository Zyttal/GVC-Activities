import cv2
import numpy as np
import activity1
import matplotlib.pyplot as plt

if __name__ == "__main__":
    image_used = "panda1.jpg"
    other_image = "panda2.jpg"

    img1 = cv2.imread(image_used, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(other_image, cv2.IMREAD_UNCHANGED)

    # Original Image
    img1 = activity1.resizeImage(img1)
    img2 = activity1.resizeImage(img2)

    images = np.concatenate((img1, img2), axis=1)

    cv2.imshow("Images Involved", images)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Histogram of both images
    img1 = cv2.imread(image_used, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(other_image, 0)

    histogram1 = cv2.calcHist([img1],[0],None,[256],[0,256]) 
    histogram2 = cv2.calcHist([img2],[0],None,[256],[0,256]) 

    plt.plot(histogram1, label="Image 1")
    plt.plot(histogram2, label="Image 2")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.title("Histograms of Image Intensities")
    plt.legend()

    plt.tight_layout()
    plt.show()