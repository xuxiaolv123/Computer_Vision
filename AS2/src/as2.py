#CS512 HW2 Li Xu A20300818

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import math

#read image
if len(sys.argv) == 2:
    source_image = cv2.imread(sys.argv[1])
elif len(sys.argv) < 2:
    cap = cv2.VideoCapture(0)
    retval, image = cap.read()

count = 0


def nothing(x):
    pass

def func_c(image, count):
    '''temp = image
    if count == 0:
        temp[:,:,2] = 0
    elif count == 1:
        temp[:,:,1] = 0
    elif count == 2:
        temp[:,:,0] = 0
    return temp'''
    temp = np.zeros((len(image),len(image[0]),3),np.uint8)
    for i in range(len(image)):
        for j in range(len(image[1])):
            if count == 0:
                temp[i,j] = [image[i,j,0],0,0]
            elif count == 1:
                temp[i,j] = [0,image[i,j,1],0]
            elif count == 2:
                temp[i,j] = [0,0,image[i,j,2]]
            pass
    return temp

def myownfunction(image,kernel):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return cv2.filter2D(image,-1,kernel)

#create track bar
cv2.namedWindow('CS512 HW2')
cv2.createTrackbar('Smoothing','CS512 HW2',2,50,nothing)
cv2.createTrackbar('Pixels','CS512 HW2',10,100,nothing)
cv2.createTrackbar('Angle','CS512 HW2',0,360,nothing)
cv2.createTrackbar('Length','CS512 HW2',5,10,nothing)

'''# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)'''



print 'Press h for help'
image = source_image

while(1):
    cv2.imshow('CS512 HW2', image)
    h,w = image.shape[:2]   #get height and width of the image

    key = cv2.waitKey(10) & 0xFF
    
    #press ESC to quit
    if key == 27:
        cv2.destroyAllWindows()
        break

    if key == ord('i'):
        print 'reload the original image'
        image = source_image

    elif key == ord('w'):
        print 'save the current image into the file out.jpg'
        cv2.imwrite('out.jpg', image)

    elif key == ord('g'):
        print 'convert the image to grayscale using OpenCV conversion function'
        if len(image.shape) != 3:
            print 'already in grayscale'
            continue
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image_bw

    elif key == ord('G'):
        print 'convert the image to grayscale using your implementation of conversion function'
        if len(image.shape) != 3:
            print 'already in grayscale'
            continue
        outbwimg = np.ones((image.shape[0], image.shape[1]), np.uint8)
        for i in range(0, image.shape[0], 1):
            for j in range(0, image.shape[1], 1):
                R,G,B = image[i,j]
                gray = 0.21 * R + 0.72 * G + 0.07 * B
                outbwimg[i,j] = gray
        image = outbwimg

    elif key == ord('c'):
        print 'cycle through the color counts of the image showing a different count every time the key is pressed'
        image = source_image
        image = func_c(image, count)
        #print count
        #print image
        count += 1
        if count >= 3:
            count = 0
    
    elif key == ord('s'):
        print 'convert the image to grayscale and smooth it using the openCV function, use a track bar to control the amount of smoothing'
        if len(image.shape) != 3:
            print 'already in grayscale'
            continue
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #cv2.imshow('CS512 HW2',image)
        smoothing = cv2.getTrackbarPos('Smoothing','CS512 HW2')
        kernel = np.ones((smoothing * smoothing),np.float32) / (smoothing * smoothing)
        dst = cv2.filter2D(image,-1,kernel)
        image = dst
        cv2.imshow('CS512 HW2',image)

    elif key == ord('S'):
        print 'convert the image to grayscale and smooth it using my own function, use a track bar to control the amount of smoothing'
        smoothing = cv2.getTrackbarPos('Smoothing','CS512 HW2')
        kernel = np.ones((smoothing * smoothing),np.float32) / (smoothing * smoothing)
        image = myownfunction(image,kernel)
        cv2.imshow('CS512 HW2',image)

    elif key == ord('d'):
        print 'downsample the image by a factor of 2 without smoothing'
        image = cv2.pyrDown(image,dstsize = (w/2,h/2))
        cv2.imshow('CS512 HW2',image)

    elif key == ord('D'):
        print 'downsample the image by a factor of 2 with smoothing'
        #smoothing = cv2.getTrackbarPos('Smoothing','CS512 HW2')
        smoothing = 5
        kernel = np.ones((smoothing * smoothing),np.float32) / (smoothing * smoothing)
        dst = cv2.filter2D(image,-1,kernel)
        image = dst
        image = cv2.pyrDown(image,dstsize = (w/2,h/2))
        cv2.imshow('CS512 HW2',image)

    elif key == ord('x'):
        print 'convert the image to grayscale and perform convolution with an x derivative filter'
        temp = np.ones((image.shape[0], image.shape[1]), np.uint8)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        for i in range(len(temp)-1):
            for j in range(len(temp[i])):
                temp[i,j] = image[i+1,j] - image[i,j]   #derivative of x = delta(x+1,y) - delta(x,y)
        image = temp
        '''image = cv2.GaussianBlur(image,(5,5),0)
        print len(image.shape)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        else:
            print 'already in grayscale'
        sobelx = cv2.Sobel(image,cv2.CV_16S,1,0,ksize = 5)
        abs_sobelx = cv2.convertScaleAbs(sobelx)
        cv2.imshow('CS512 HW2',abs_sobelx)'''
        '''sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
        plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
        plt.show()'''

    elif key == ord('y'):
        print 'convert the image to grayscale and perform convolution with an y derivative filter'
        temp = np.ones((image.shape[0], image.shape[1]), np.uint8)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        for i in range(len(temp)):
            for j in range(len(temp[i])-1):
                temp[i,j] = image[i,j+1] - image[i,j]   #derivative of y = delta(x,y+1) - delta(x,y)
        image = temp

    elif key == ord('m'):
        print 'show the magnitude of the gradient normalized to the range [0,255]'
        #use sobel filter to compute magnitute of gradient
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray,cv2.CV_16S,1,0,ksize = 3)
        grad_y = cv2.Sobel(gray,cv2.CV_16S,0,1,ksize = 3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)   # converting back to uint8
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        image = cv2.addWeighted(abs_grad_x,0.5,abs_grad_y,0.5,0)    #It's only the approximate solution
        #image = math.sqrt(abs_grad_x * abs_grad_x + abs_grad_y * abs_grad_y)
        cv2.imshow('CS512 HW2',image)

    elif key == ord('p'):
        print 'convert the image to grayscale and plot the gradient vectors of the image every N pixels and let the plotted gradient vectors have a length of K'
        #Use function x and y to compute gradient first
        temp = np.ones((image.shape[0], image.shape[1],3), np.uint8)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        for i in range(len(temp)-1):
            for j in range(len(temp[i])-1):
                dx = image[i+1,j] - image[i,j]
                dy = image[i,j+1] - image[i,j]
                norm = math.sqrt(dx * dx + dy * dy) / math.sqrt(255 * 255 + 255 * 255)
                temp[i,j] = [dx,dy,norm]

        #get N pixels value and K length value from trackbar
        pixels = cv2.getTrackbarPos('Pixels','CS512 HW2')
        K = cv2.getTrackbarPos('Length','CS512 HW2')

        i = pixels
        for i in xrange(i,len(temp),pixels):
            j = pixels
            for j in xrange(j,len(temp[i]),pixels):
                [dx,dy,norm] = temp[i,j]
                if math.isnan(dy/dx):
                    continue
                elif (dy/dx <= 1) and (dy/dx >= -1) and dx >= 0:
                    cv2.line(image,(i,j),(i+K,int(j+K*dy/dx)),(255,0,0),1)
                elif (dy/dx <= 1) and (dy/dx >= -1) and dx < 0:
                    cv2.line(image,(i,j),(i-K,int(j-K*dy/dx)),(255,0,0),1)
                elif (dy/dx > 1) and (dy/dx < -1) and dy >= 0:
                    cv2.line(image,(i,j),(int(i+K*dx/dy),int(j+K)),(255,0,0),1)
                elif (dy/dx > 1) and (dy/dx < -1) and dy < 0:
                    cv2.line(image,(i,j),(int(i-K*dx/dy),int(j-K)),(255,0,0),1)
        cv2.imshow('CS512 HW2',image)

    elif key == ord('r'):
        print 'convert the image to grayscale and rotate it using an angle of Q degrees. Use a track bar to control the rotation angle'
        if len(image.shape) == 3:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        rows = image.shape[0]
        cols = image.shape[1]
        angle = cv2.getTrackbarPos('Angle','CS512 HW2')
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        image = cv2.warpAffine(image,M,(cols,rows))
        cv2.imshow('CS512 HW2',image)

    elif key == ord('h'):
        print """
        Press keys below to call functions
        
        i   reload the original image
        
        w   save the current image into the file 'out.jpg'
        
        g   convert the image to grayscale using the openCV conversion function
        
        G   convert the image to grayscale using your implementation of conversion function
        
        c   cycle through the color channels of the image showing a different channel every time the key is pressed
        
        s   convert the image to grayscale and smooth it using the openCV function
        
        S   convert the image to grayscale and smooth it using your function which should perform convolution with a suitable filter
        
        d   downsample the image by factor of 2 without smoothing
        
        D   downsample the image by factor of 2 with smoothing
        
        x   convert the image to grayscale and perform convolution with an x derivative filter
        
        y   convert the image to grayscale and perform convolution with a y derivative filter
        
        m   show the magnitude of the gradient normalized to the range [0,255]
        
        p   convert the image to grayscale and plot the gradient vectors of the image every N pixels and let the plotted gradient vectors have a length of K
        
        r   convert the image to grayscale and rotate it using an angle of Q degrees
        
        h   help
        """






