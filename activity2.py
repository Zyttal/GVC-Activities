import cv2
import numpy as py
import activity1

if __name__ == "__main__":
    image_used = "panda1.jpg"
    img = cv2.imread(image_used, cv2.IMREAD_COLOR)

    print("COLORED IMAGE PROPERTIES")
    activity1.printImageProperties(img)

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

    dim = (1024, 768)
    resized_img = cv2.resize(canny_edges, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("Canny Edge Detection", resized_img)

    cv2.waitKey(0)

    '''
        Sobel Edge Detection is a popular edge detection method that is based on the concept of finding the gradient of the intensity of the image. It is simple, efficient, and widely used in image processing applications, including object recognition, image segmentation, and pattern recognition.
    '''

    # Repeat steps of grayscale and Gaussian blur then apply Sobel Algorithm from cv2
    sobelx = cv2.Sobel(src=blur_img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) 
    sobely = cv2.Sobel(src=blur_img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) 
    sobelxy = cv2.Sobel(src=blur_img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

    sobelx = cv2.resize(sobelx, dim, interpolation = cv2.INTER_AREA)
    sobely = cv2.resize(sobelx, dim, interpolation = cv2.INTER_AREA)
    sobelxy = cv2.resize(sobelx, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('Sobel X', sobelx)
    cv2.waitKey(0)

    cv2.imshow('Sobel Y', sobely)
    cv2.waitKey(0)

    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Display RGB of Image separately 

    (R, G, B) = cv2.split(img)

    R = cv2.resize(R, dim, interpolation = cv2.INTER_AREA)
    G = cv2.resize(G, dim, interpolation = cv2.INTER_AREA)
    B = cv2.resize(B, dim, interpolation = cv2.INTER_AREA)
    
    cv2.imshow('Red Value of Image', R)
    cv2.imshow('Blue Value of Image', B)
    cv2.imshow('Green Value of Image', G)
    cv2.waitKey(0)

    cv2.destroyAllWindows()