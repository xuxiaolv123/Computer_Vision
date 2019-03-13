import cv2
import numpy as np
import sys
import math


count = 0

def norm(points):
	norm_points = []
	sum_x = 0
	sum_y = 0
	for point in points:
		sum_x += point[1]
		sum_y += point[0]
	mean_x = sum_x/len(points)
	mean_y = sum_y/len(points)
	for point in points:
		stdv_x = math.sqrt((point[1] - mean_x)**2)
		stdv_y = math.sqrt((point[0] - mean_y)**2)
	for point in points:
		x = (point[1] - mean_x) / stdv_x
		y = (point[0] - mean_y) / stdv_y
		norm_points.append([y,x])
	return mean_x, mean_y, math.sqrt((stdv_x)**2 + (stdv_y)**2), norm_points

def cal_epipole(fmatrix, isleft = True):
	if isleft:
		s,d,v = np.linalg.svd(fmatrix)
		left_epipole = v[-1][0:2]	#left epipole is the right null space of F
		return left_epipole
	elif not isleft:
		s,d,v = np.linalg.svd(np.transpose(fmatrix))
		right_epipole = v[-1][0:2]	#right epipole is the left null space of F
		return right_epipole


def draw_rightepipolarline(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
			cv2.circle(image_l,(x,y),10,(255,0,0),0)
			temp = np.dot(Fmatrix ,np.array([x,y,1]))
			# y=kx+b
			k = (-1)*temp[0]/temp[1]
			b = (-1)*temp[2]/temp[1]
			x_1 = 0
			y_1 = int(b)
			x_2 = len(image_r[0])
			y_2 = int(k*x_2 +b)

			#print x_1,y_1, x_2,y_2

			cv2.line(image_r, (x_1,y_1), (x_2,y_2), (255,0,0))
			cv2.imshow("right image", image_r)

def draw_leftepipolarline(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
		cv2.circle(image_r,(x,y),10,(255,0,0),0)
		temp = np.dot(np.transpose(Fmatrix) ,np.array([x,y,1]))
		# y=kx+b
		k = (-1)*temp[0]/temp[1]
		b = (-1)*temp[2]/temp[1]
		x_1 = 0
		y_1 = int(b)
		x_2 = len(image_l[0])
		y_2 = int(k*x_2 +b)        

		cv2.line(image_l, (x_1,y_1), (x_2,y_2), (255,0,0))
		cv2.imshow("left image", image_l)

def cal_fmatrix(leftpoints, rightpoints):
	A = []

	for i in range(len(leftpoints)):
		[y_left,x_left] = leftpoints[i]
		[y_rihgt,x_right] = rightpoints[i]
		A.append([x_left*x_right, x_right*y_left, x_right, y_rihgt*x_left, y_rihgt*y_left, y_rihgt, x_left, y_left, 1])
	Amatrix = np.array(A)


	dummy1, dummy2, v = np.linalg.svd(Amatrix)

	f = v[-1] #take the last col of V, which is the right null space
	f = f/f[-1]

	fmatrix = np.reshape(f,(3,3))	#reshape F to 3x3 matrix

	s,d,v = np.linalg.svd(fmatrix)	# to ensure F is a rank 2 matrix, do svd again

	d_prime = np.dot(np.diag(d), np.diag(np.array([1,1,0])))	#get d_prime, take d withg last diagonal element is 0

	fmatrix_prime = np.dot(np.dot(s,d_prime), v)	#F_prime is rank 2

	return fmatrix_prime


def draw_circle(event,x,y,flags,param):
	global isleft
	global count
	if event == cv2.EVENT_LBUTTONDOWN:
		if isleft == 1:
			cv2.circle(image_l,(x,y),10,(0,0,255),0)
			leftpoints.append([y,x])
			isleft = 0
		elif isleft == 0:
			cv2.circle(image_r,(x,y),10,(0,0,255),0)
			rightpoints.append([y,x])
			isleft = 1
			count += 1

def choosepoints():
	global isleft
	cv2.namedWindow('choose corresponding points')
	cv2.setMouseCallback('choose corresponding points',draw_circle)
	while count < 8:	#we only select 9 points, it should be more accurate if we choose more points
		if isleft == 1:
			cv2.imshow('choose corresponding points', image_l)
		elif isleft == 0:
			cv2.imshow('choose corresponding points', image_r)
		if cv2.waitKey(20) & 0xFF == 27:
			break


if len(sys.argv) < 2:
	print '''
Instruction: Run this program with arguments: python as5.py left.jpg right.jpg
	'''
else:
	image_l = cv2.imread(sys.argv[1])
	image_r = cv2.imread(sys.argv[2])
	cv2.imshow('left image', image_l)
	cv2.imshow('right image', image_r)
	print 'Press any key to choose corresponding points'
	key = cv2.waitKey(0)
	if key == 27:
		cv2.destroyAllWindows()

	print 'Please choose the 8 pairs of corresponding points on the images\n'

	image_l = cv2.imread(sys.argv[1])
	image_r = cv2.imread(sys.argv[2])


	leftpoints = []
	rightpoints = []
	isleft = 1
	choosepoints()
	#print 'corresponding points in left image: \n', leftpoints
	#print 'corresponding points in right image: \n', rightpoints

	for index, point in enumerate(leftpoints):
		cv2.circle(image_l, (point[1],point[0]), 10, (0,0,255), 0)
		cv2.putText(image_l, str(index+1), (point[1],point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)

	for index, point in enumerate(rightpoints):
		cv2.circle(image_r, (point[1],point[0]), 10, (0,0,255), 0)
		cv2.putText(image_r, str(index+1), (point[1],point[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)

	cv2.imshow('left image with corresponding points', image_l)
	cv2.imshow('right image with corresponding points', image_r)

	print 'Press any key to compute fundamental matrix and epipoles\n'

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	#normalize points
	mean_x_left, mean_y_left, stdv_left, norm_leftpoints = norm(leftpoints)
	mean_x_right, mean_y_right, stdv_right, norm_rightpoints = norm(rightpoints)

	#print 'norm left is ', norm_leftpoints
	#print 'norm right is ', norm_rightpoints

	Fmatrix_prime = cal_fmatrix(norm_leftpoints,norm_rightpoints)

	M = np.array([[1/stdv_right, 0, -mean_x_right],[0, 1/stdv_right, -mean_y_right],[0,0,1]])
	M_prime = np.array([[1/stdv_left, 0, -mean_x_left],[0, 1/stdv_left, -mean_y_left],[0,0,1]])

	Fmatrix = np.dot(np.transpose(M), np.dot(Fmatrix_prime, M_prime))	# de-normalize F

	#
	Fmatrix = cal_fmatrix(leftpoints,rightpoints)

	print 'F matrix is: \n', Fmatrix

	left_epipole = cal_epipole(Fmatrix, True)
	print 'left epipole is: ', left_epipole
	right_epipole = cal_epipole(Fmatrix, False)
	print 'right epipole is: ', right_epipole

	while(1):
		print 'Press l (left) to choose a point in left image and draw the corresponding epipolar line in the right image\n'
		print 'Press r (right) to choose a point in right image and draw the corresponding epipolar line in the left image\n'

		image_l = cv2.imread(sys.argv[1])
		image_r = cv2.imread(sys.argv[2])
		cv2.imshow('left image', image_l)
		cv2.imshow('right image', image_r)

		key = cv2.waitKey(0)

		if key == ord('l'):
			isleft = 1
			cv2.namedWindow("choose a point in left image")
			cv2.setMouseCallback("choose a point in left image", draw_rightepipolarline)
			print '\nSelect ponits to draw epipolar lines or Press ESC to draw epipolar lines in another image\n'
			while(1):
				cv2.imshow("choose a point in left image", image_l)
				if cv2.waitKey(20) & 0xFF == 27:
					cv2.destroyAllWindows()
					break

		elif key == ord('r'):
			isleft == 0
			cv2.namedWindow("choose a point in right image")
			cv2.setMouseCallback("choose a point in right image", draw_leftepipolarline)
			print '\nSelect points to draw epipolar lines Press ESC to draw epipolar lines in another image\n'
			while(1):
				cv2.imshow("choose a point in right image", image_r)
				if cv2.waitKey(20) & 0xFF == 27:
					cv2.destroyAllWindows()
					break

		elif key == 27:
			break
