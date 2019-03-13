import cv2
import numpy as np
import sys


def calflow(image1,image2,a,threshold):
	u_ini = np.zeros([image1.shape[0],image1.shape[1]])
	v_ini = np.zeros([image2.shape[0],image2.shape[1]])

	u = u_ini
	v = v_ini

	[fx, fy, ft] = computeDerivatives(image1,image2)
	fx = xDer(image1, image2)
	fy = yDer(image1, image2)
	ft = tDer(image1, image2)

	hskernel = np.matrix([[1/12, 1/6, 1/12],[1/6, 0, 1/6],[1/12, 1/6, 1/12]])	#laplacian kernel

	#Since it's hard to find a threshold, I just iterate a fixed number of time to find the fina u,v
	for i in range(threshold):
		u0 = np.copy(u)
		v0 = np.copy(v)

		uAvg = cv2.filter2D(u0.astype(np.float32), -1, hskernel)
		vAvg = cv2.filter2D(v0.astype(np.float32), -1, hskernel)


		u = uAvg - 1.0/(a**2 + fx**2 + fy**2) * fx * (fx * uAvg + fy * vAvg + ft)
		v = vAvg - 1.0/(a**2 + fx**2 + fy**2) * fy * (fx * uAvg + fy * vAvg + ft)

	res = [[] for i in range(len(v))]
	for i in range(0,len(v),1):
		for j in range(0,len(v[i]),1):
			res[i].append([u[i][j], v[i][j]])
	flow = np.asarray(res)
        
	return flow

def takePixel(img, i, j):
	i = i if i >= 0 else 0
	j = j if j >= 0 else 0
	i = i if i < img.shape[0] else img.shape[0] - 1
	j = j if j < img.shape[1] else img.shape[1] - 1
	return img[i, j]


def xDer(img1, img2):
	res = np.zeros_like(img1)
	for i in xrange(res.shape[0]):
		for j in xrange(res.shape[1]):
			sm = 0
			sm += takePixel(img1, i,     j + 1) - takePixel(img1, i,     j)
			sm += takePixel(img1, i + 1, j + 1) - takePixel(img1, i + 1, j)
			sm += takePixel(img2, i,     j + 1) - takePixel(img2, i,     j)
			sm += takePixel(img2, i + 1, j + 1) - takePixel(img2, i + 1, j)
			sm /= 4.0
			res[i, j] = sm
	return res

def yDer(img1, img2):
	res = np.zeros_like(img1)
	for i in xrange(res.shape[0]):
		for j in xrange(res.shape[1]):
			sm = 0
			sm += takePixel(img1, i + 1, j    ) - takePixel(img1, i, j    )
			sm += takePixel(img1, i + 1, j + 1) - takePixel(img1, i, j + 1)
			sm += takePixel(img2, i + 1, j    ) - takePixel(img2, i, j    )
			sm += takePixel(img2, i + 1, j + 1) - takePixel(img2, i, j + 1)
			sm /= 4.0
			res[i, j] = sm
	return res

def tDer(img, img2):
	res = np.zeros_like(img)
	for i in xrange(res.shape[0]):
		for j in xrange(res.shape[1]):
			sm = 0
			for ii in xrange(i, i + 2):
				for jj in xrange(j, j + 2):
					sm += takePixel(img2, ii, jj) - takePixel(img, ii, jj)
			sm /= 4.0
			res[i, j] = sm
	return res

def computeDerivatives(image1,image2):
	# build kernels for calculating derivatives
	kernelX = np.matrix([[-1,1],[-1,1]])*.25 #kernel for computing dx
	kernelY = np.matrix([[-1,-1],[1,1]])*.25 #kernel for computing dy
	kernelT = np.ones([2,2])*.25

    #apply the filter to every pixel using OpenCV's convolution function
	fx = cv2.filter2D(image1,-1,kernelX) + cv2.filter2D(image2,-1,kernelX)
	fy = cv2.filter2D(image1,-1,kernelY) + cv2.filter2D(image2,-1,kernelY)
    # ft = im2 - im1
	ft = cv2.filter2D(image2,-1,kernelT) + cv2.filter2D(image1,-1,-kernelT)
	return (fx,fy,ft)

def draw_flow(img, flow, step=16):
	h, w = img.shape[:2]
	y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
	fx, fy = flow[y,x].T
	lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
	lines = np.int32(lines + 0.5)
	vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	cv2.polylines(vis, lines, 0, (0, 255, 0))
	for (x1, y1), (_x2, _y2) in lines:
	    cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
	return vis

if len(sys.argv) < 2:
    print '''
Instruction: Run this program with arguments: python as6.py test.mpg
    '''
else:
	print '1. Press P to pause and resume the video.'
	print '2.Press ESC to quit'
	print 'Due to the large calculation, please be patient OTZ\n'

	cam = cv2.VideoCapture(sys.argv[1])
	ret, prev = cam.read()
	prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
	prevgray = prevgray[:,:240]


while(1):
    ret, img = cam.read()
    if ret == True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray[:,:240]
        flow = calflow(prevgray,gray,1,100)
        prevgray = gray
        cv2.imshow('Optical Flow', draw_flow(gray,flow))


        ch = 0xFF & cv2.waitKey(5)
        if ch == ord('p'):
        	#cv2.putText(gray, 'PAUSED', (120,120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
        	key = 0xFF & cv2.waitKey(-1)	#This part I set the dalay value of waitKey function to a negtive number(-1), when p is pressed. the video got paused until another key is pressed
        if ch == 27:
            break
    else:
        break
cv2.destroyAllWindows()
cam.release()