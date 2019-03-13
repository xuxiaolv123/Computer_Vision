CS512 - AS1 - Li Xu - A20300818

All materials are under one folder as Professor said on discussion board in Blackboard
 
Answers to all questions are in as1.pdf

as1.py is coded in Python 2.7.14 and Numpy 1.13.1

In Terminal or Command Prompt run as1.py using command: python as1.py

The result shows as below:

C:\Users\Li Xu\repos\cs512-f17-li-xu\AS1>python as1.py

  This is problem A:

1. 2A - B =  [-2 -1  0]

2. ||A|| =  3.74165738677

The angle between A and X-axis in degree:  74.4986404331

3. The unit vetor in the direction A:  [ 0.26726124  0.53452248  0.80178373]

4. The directional cosines of A:  0.267261241912 0.534522483825 0.801783725737

5. A*B =  32  B*A =  32

6. The angle between A and B in degree:  12.9331544919

7. A vector which is perpendicular to A:  [-3  0  1]

8. A X B =  [-3  6 -3]  B X A =  [ 3 -6  3]

9. A vector which is perpendicular to both A and B:  [-3  6 -3]

10. linear dependency between A, B, C: 3A -1B = C

11. A^TB =  32 
AB^T = 
[[ 4  5  6]
 [ 8 10 12]
 [12 15 18]]
##########################################################

This is problem B:

1. 2A - B = 
[[ 1  2  5]
 [ 6 -5 10]
 [-3 12 -3]]

2. AB = 
[[ 14  -2  -4]
 [  9   0  15]
 [  7   7 -21]] 
BA = 
[[  9   3   8]
 [  6 -18  13]
 [ -5  15   2]]

3. (AB)^T = 
[[ 14   9   7]
 [ -2   0   7]
 [ -4  15 -21]] 
B^TA^T = 
[[ 14   9   7]
 [ -2   0   7]
 [ -4  15 -21]]

4. |A| = 
55.0 
|C| = 
0.0

 5. The row vectors form an orthogonal set:

A is  False

B is  True

C is  False

6. A^-1 = 
[[-0.23636364  0.30909091  0.21818182]
 [ 0.07272727 -0.01818182  0.16363636]
 [ 0.36363636 -0.09090909 -0.18181818]] 
B^-1 = 
[[ 0.16666667  0.0952381   0.21428571]
 [ 0.33333333  0.04761905 -0.14285714]
 [ 0.16666667 -0.19047619  0.07142857]]
##########################################################

This is problem C

1.The eigenvalues and corresponding eigenvectors of A:
(array([-1.,  4.]), matrix([[-0.70710678, -0.5547002 ],
        [ 0.70710678, -0.83205029]]))
2. The matrix V-1AV where V is composed of the eigenvectors of A:
[[-1. -0.]
 [ 0.  4.]]

3. The dot product between the eigenvectors of A
[[-0.19611614]]

4. The dot product between the eigenvectors of A
[[ 0.]]

5. The eigenvectors of B are orthogonal because of B is a symmetric matrix