# -*- coding: UTF-8 -*-

'''CS512 HW3 Li Xu A20300818'''

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

def nothing(x):
    pass

def harris(image,variance,traceweight):
    h = image.shape[0]
    w = image.shape[1]

    #1st deriative
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, 5)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, 5)

    # Variance of Gaussian can only be odd value in cv2.GaussianBlur function
    ksize = max(5, (int)(5 * variance))
    if (ksize % 2 == 0):
        ksize = ksize + 1

    dxx = cv2.GaussianBlur(dx ** 2, (ksize, ksize), variance)
    dyy = cv2.GaussianBlur(dy ** 2, (ksize, ksize), variance)
    dxy = cv2.GaussianBlur(dx * dy, (ksize, ksize), variance)

    dst = np.zeros((h, w), dtype=np.float32)
    for i in range(0, h):
        for j in range(0, w):
            Sxx = dxx[i, j]
            Sxy = dxy[i, j]
            Syy = dyy[i, j]
            det = Sxx * Syy - Sxy * Sxy
            trace = Sxx + Syy
            dst[i, j] = det - traceweight * (trace * trace)
    return dst

def harrisDetection(image,variance,size,traceweight,threshold):
    #covert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dst = harris(gray,variance,traceweight)
    #dst3 = cv2.cornerHarris(gray, size, variance, threshold)
    ret, dst1 = cv2.threshold(dst, 0.01 * threshold * dst.max(), 255, 0)

    # find centroids of the points around the area
    dst2 = np.uint8(dst1)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst2)

    # localization of corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    return  corners

def drawRec(image1, image2, corners1,corners2,size):
    # covert image to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    corners1 = np.int0(corners1)
    corners2 = np.int0(corners2)

    # draw a rectangle around the corner
    for i in range(0, corners1.shape[0]):
        cv2.rectangle(gray1, (corners1[i, 0] - size, corners1[i, 1] - size),
                      (corners1[i, 0] + size, corners1[i, 1] + size), (0,0,255))
    for j in range(0,corners2.shape[0]) :
          cv2.rectangle(gray2, (corners2[j,0] - size, corners2[j,1]-size),
                        (corners2[j,0]+size, corners2[j,1]+size), (0,0,255))

    cv2.imshow('corner position1', gray1)
    cv2.imshow('corner position2', gray2)


def wrapper(img1,img2):

    cv2.namedWindow('trackbar')
    trace_bar = np.zeros((300, 500), dtype=np.float32)
    cv2.createTrackbar('Variance of Gaussian', 'trackbar', 0, 10, nothing)
    cv2.createTrackbar('Neighborhood Size', 'trackbar', 5, 10, nothing)
    cv2.createTrackbar('Weight of trace', 'trackbar', 4, 15, nothing)
    cv2.createTrackbar('Threshold value', 'trackbar', 10, 50, nothing)
    cv2.imshow('trackbar', trace_bar)

    while (1):
        key = cv2.waitKey(10) & 0xFF

        # press ESC to quit
        if key == 27:
            cv2.destroyAllWindows()
            break

        if key == ord('r'):
            print 'Reset the images'
            image1 = cv2.imread(sys.argv[1])
            image2 = cv2.imread(sys.argv[2])
            wrapper(image1,image2)

        if key == ord('h'):
            print 'Harris Line Detection'

            image1 = img1.copy()
            image2 = img2.copy()

            variance = cv2.getTrackbarPos('Variance of Gaussian','trackbar')
            size = cv2.getTrackbarPos('Neighborhood Size','trackbar')
            traceweight = cv2.getTrackbarPos('Weight of trace','trackbar')
            threshold = cv2.getTrackbarPos('Threshold value','trackbar')


            #Because trace is usually selected in [0.04,0.15]
            traceweight = traceweight * 0.01


            #Harris corner detection and localization
            corner1 = harrisDetection(image1,variance,size,traceweight,threshold)
            corner2 = harrisDetection(image2,variance,size,traceweight,threshold)

            #Draw Rectangles over each corner
            drawRec(image1, image2, corner1, corner2,size)

            #Number the corresponding corner using cv2.orb function
            #Since there are too many same feature, I only number the first 50 corners
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(image1, None)
            kp2, des2 = orb.detectAndCompute(image2, None)
            # create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors.
            matches = bf.match(des1, des2)

            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)
            #print len(matches)

            list_kp1 = []
            list_kp2 = []

            for mat in matches:
                # Get the matching keypoints for each of the images
                img1_idx = mat.queryIdx
                img2_idx = mat.trainIdx

                # x - columns
                # y - rows
                # Get the coordinates
                (x1, y1) = kp1[img1_idx].pt
                (x2, y2) = kp2[img2_idx].pt

                # Append to each list
                list_kp1.append((x1, y1))
                list_kp2.append((x2, y2))
            #print list_kp1[0]

            #img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches,None, flags=2)

            for i in range(0,50):
                point1 = list_kp1[i]
                point2 = list_kp2[i]
                cv2.putText(image1, str(i),(int(point1[0]),int(point1[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
                cv2.putText(image2, str(i), (int(point2[0]),int(point2[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
            cv2.imshow('feature1', image1)
            cv2.imshow('feature2', image2)

            #plt.imshow(img3), plt.show()




def main():
    if len(sys.argv) < 2:
        print '''
            Instruction: Pass two images (python as3.py img1.jpg img2.jpg) or capture from camera (python as3.py camera)
            After load images, press h to do harris corner detection
            '''
    elif 'camera' == sys.argv[1]:
        print 'Press Spcae to capture image from camera'
        cap = cv2.VideoCapture(0)
        img_counter = 1
        while(1):
            ret, frame = cap.read()
            cv2.imshow('CS512 HW3',frame)
            if cv2.waitKey(10) == 27 :
                cv2.destroyAllWindows()
                break
            if cv2.waitKey(10) == 32 :
                # SPACE pressed
                img_name = "camimg{}.jpg".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1
        cam.release()
        cv2.destroyAllWindows()
    else:
        print 'Press h to do harris corner detection'
        image1 = cv2.imread(sys.argv[1])
        image2 = cv2.imread(sys.argv[2])
        cv2.imshow('image1',image1)
        cv2.imshow('image2',image2)

        img1 = image1.copy()
        img2 = image2.copy()

        wrapper(img1,img2)


if __name__ == '__main__':
    main()
