import cv2
import numpy as np


def segment_image(image_path, thickness=2):
    # Reading the image
    img = cv2.imread(image_path)
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)


    # Converting the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscaled image', gray)
    cv2.waitKey(0)


    # Applying threshold to create a binary image
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('Thresholded Binary Image', thresh)
    cv2.waitKey(0)


    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow('After Noise Removal', opening)
    cv2.waitKey(0)


    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    cv2.imshow('Sure Background', sure_bg)
    cv2.waitKey(0)



    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    cv2.imshow('Sure Foreground', sure_fg)
    cv2.waitKey(0)



    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imshow('Unknown Region', unknown)
    cv2.waitKey(0)



    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Mark the region of unknown with 0
    markers[unknown == 255] = 0


    # Apply the watershed algorithm with increased thickness
    cv2.watershed(img, markers)

    # Draw thicker marker boundaries (use color [0, 255, 0] for green)
    img[markers == -1] = [0, 255, 0]  # Mark watershed boundaries with green color

    return img


def main():
   
    image_path = '/Users/sasidhar/Downloads/combined46131.jpg'

    # Perform image segmentation using the watershed algorithm with increased thickness
    segmented_image = segment_image(image_path, thickness=100)

    # Display the original and segmented images
    original_image = cv2.imread(image_path)


    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()