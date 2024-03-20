import cv2
import numpy as np
import random
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import os
from PIL import Image

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

def main():
    whileRunning = True

    while(whileRunning):
        os.system('clear')

        print("GVC Activities 1 - 5 Compilation")
        print("1 - Activity #1")
        print("2 - Activity #2")
        print("3 - Activity #3")
        print("4 - Activity #4")
        print("5 - Activity #5")
        print("6 - Exit Program")

        option = int(input("Enter Option (1-6): "))
        match option:
            case 1:
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
            case 2:
                image_used = "panda1.jpg"
                img = cv2.imread(image_used, cv2.IMREAD_COLOR)

                print("COLORED IMAGE PROPERTIES")
                printImageProperties(img)

                # Get 4 different edges of an image (Use different parameters/method)

                '''
                    Canny is a multi-stage edge detector. It uses a filter based on the derivative of a Gaussian in order to compute the intensity of the gradients.The Gaussian reduces the effect of noise present in the image. Then, potential edges are thinned down to 1-pixel curves by removing non-maximum pixels of the gradient magnitude.
                '''

                # Grayscale the image 
                img_gray = cv2.imread(image_used, cv2.IMREAD_GRAYSCALE)

                # Apply Gaussian Blur for better edge detection
                blur_img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)

                # Thresholding and the other steps are applied
                canny_edges = cv2.Canny(image=blur_img_gray, threshold1=100, threshold2=200)

                canny_edges = resizeImage(canny_edges)
                
                cv2.imshow("Canny Edge Detection", canny_edges)

                cv2.waitKey(0)
                cv2.destroyAllWindows()

                '''
                    Sobel Edge Detection is a popular edge detection method that is based on the concept of finding the gradient of the intensity of the image. It is simple, efficient, and widely used in image processing applications, including object recognition, image segmentation, and pattern recognition.
                '''

                # Repeat steps of grayscale and Gaussian blur then apply Sobel Algorithm from cv2
                sobelxy = cv2.Sobel(src=blur_img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) 

                sobelxy = resizeImage(sobelxy)

                cv2.imshow('Sobel X Y using Sobel() function', sobelxy)

                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Display RGB of Image separately 

                (R, G, B) = cv2.split(img)

                R = resizeImage(R)
                G = resizeImage(G)
                B = resizeImage(B)
                
                images = np.concatenate((R,G,B), axis=1)

                cv2.imshow('R G B (Left to Right) Channels', images)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            case 3:
                image_used = "panda1.jpg"
                other_image = "panda2.jpeg"

                img1 = cv2.imread(image_used, cv2.IMREAD_UNCHANGED)
                img2 = cv2.imread(other_image, cv2.IMREAD_UNCHANGED)

                # Original Image
                img1 = resizeImage(img1)
                img2 = resizeImage(img2)

                images = np.concatenate((img1, img2), axis=1)

                cv2.imshow('Images', images)

                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # Histogram of both images
                img1 = cv2.imread(image_used, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(other_image, cv2.IMREAD_GRAYSCALE)

                histogram1 = cv2.calcHist([img1],[0],None,[256],[0,256])
                histogram2 = cv2.calcHist([img2],[0],None,[256],[0,256])

                plt.figure(figsize=(10, 5))

                plt.subplot(1, 2, 1)
                plt.plot(histogram1, color='b')
                plt.title("Histogram for Panda 1")
                plt.xlabel("Pixel Intensity")
                plt.ylabel("Frequency")

                plt.subplot(1, 2, 2)
                plt.plot(histogram2, color='g')
                plt.title("Histogram for Panda 2")
                plt.xlabel("Pixel Intensity")
                plt.ylabel("Frequency")

                plt.tight_layout()
                plt.show()

                img1 = cv2.imread(image_used, cv2.IMREAD_UNCHANGED)

                (R, G, B) = cv2.split(img1)

                R = resizeImage(R)
                G = resizeImage(G)
                B = resizeImage(B)

                histogramR = cv2.calcHist([R],[0],None,[256],[0,256])
                histogramG = cv2.calcHist([G],[0],None,[256],[0,256])
                histogramB = cv2.calcHist([B],[0],None,[256],[0,256])

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                axs[0].plot(histogramR, color='r')
                axs[0].set_title("Histogram for Panda 1 - R")
                axs[0].set_xlabel("Pixel Intensity")
                axs[0].set_ylabel("Frequency")

                axs[1].plot(histogramG, color='g')
                axs[1].set_title("Histogram for Panda 1 - G")
                axs[1].set_xlabel("Pixel Intensity")
                axs[1].set_ylabel("Frequency")

                axs[2].plot(histogramB, color='b')
                axs[2].set_title("Histogram for Panda 1 - B")
                axs[2].set_xlabel("Pixel Intensity")
                axs[2].set_ylabel("Frequency")

                plt.tight_layout(pad=3.0)
                plt.show()
            case 4:
                image_used = "panda1.jpg"

                img = cv2.imread(image_used, cv2.IMREAD_UNCHANGED)
                gray_img = cv2.imread(image_used, cv2.IMREAD_GRAYSCALE)
                histogram = cv2.calcHist([gray_img],[0],None,[256],[0,256])
                edges_img = cv2.Sobel(src=gray_img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) 

                fig, axs = plt.subplots(2, 2, figsize=(15,5))

                axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axs[0, 0].set_title('Original Image')
                axs[0, 0].axis('off')

                axs[0, 1].imshow(gray_img, cmap='gray')
                axs[0, 1].set_title('GrayScaled Image')
                axs[0, 1].axis('off')

                axs[1, 0].plot(histogram, color='black')
                axs[1, 0].set_title('Histogram')
                axs[1, 0].set_xlabel('Pixel Intensity')
                axs[1, 0].set_ylabel('Frequency')

                axs[1, 1].imshow(edges_img, cmap='gray')
                axs[1, 1].set_title('Edges Image')
                axs[1, 1].axis('off')

                plt.tight_layout()
                plt.show()
            case 5:
                image_used = "panda1.jpg"

                img = cv2.imread(image_used, cv2.IMREAD_UNCHANGED)
                gray_img = cv2.imread(image_used, cv2.IMREAD_GRAYSCALE)
                histogram = cv2.calcHist([gray_img],[0],None,[256],[0,256])
                edges_img = cv2.Canny(image=gray_img, threshold1=100, threshold2=200)

                fig, axs = plt.subplots(2, 2, figsize=(15,5))

                axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axs[0, 0].set_title('Original Image')
                axs[0, 0].axis('off')

                axs[0, 1].imshow(gray_img, cmap='gray')
                axs[0, 1].set_title('GrayScaled Image')
                axs[0, 1].axis('off')

                axs[1, 0].plot(histogram, color='black')
                axs[1, 0].set_title('Histogram')
                axs[1, 0].set_xlabel('Pixel Intensity')
                axs[1, 0].set_ylabel('Frequency')

                axs[1, 1].imshow(edges_img, cmap='gray')
                axs[1, 1].set_title('Edges Image')
                axs[1, 1].axis('off')

                height, width, channels = img.shape
                size_img = os.path.getsize(image_used)
                img_mode = Image.open(image_used)
                img_mode = img_mode.mode
                color_value = img[300, 300]

                print(f"Filename: {image_used}")
                print(f"Format: {img_mode}")
                print(f"Width: {width}")
                print(f"Height: {height}")
                print(f"Value of pixel [300,300]: {color_value}")

                plt.tight_layout()
                plt.show()
            case 6:
                whileRunning = False
                break
            case _:
                os.system('clear')

                print("Invalid Input!")
                input("Enter to continue...")

    print("Program has exited successfully!!")

if __name__ == "__main__":
    main()