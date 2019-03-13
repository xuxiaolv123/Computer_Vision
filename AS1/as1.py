import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6,])
c = np.array([-1,1,3])
x = np.array([1,0,0])
dot_ax = np.dot(a,x)
a_modulus = np.sqrt((a*a).sum())
x_modulus = np.sqrt((x*x).sum())
a2 = np.matrix('1,2,3;4,-2,3;0,5,-1')
b2 = np.matrix('1,2,1;2,1,-4;3,-2,1')
c2 = np.matrix('1,2,3;4,5,6;-1,1,3')
a3 = np.matrix('1,2;3,2')
b3 = np.matrix('2,-2;-2,5')

def main():
    print '\n This is problem A:'
    print '\n1. 2A - B = ', 2 * a - b
    print '\n2. ||A|| = ', np.sqrt(a.dot(a))
    print '\nThe angle between A and X-axis in degree: ', angle(a,x)
    print '\n3. The unit vetor in the direction A: ', a / mod(a)
    print '\n4. The directional cosines of A: ', 1/mod(a), 2/mod(a), 3/mod(a)
    print '\n5. A*B = ', dot(a,b), ' B*A = ', dot(b,a)
    print '\n6. The angle between A and B in degree: ', angle(a,b)
    print '\n7. A vector which is perpendicular to A: ', perpendicular_vector(a)
    print '\n8. A X B = ', cross(a,b), ' B X A = ', cross(b,a)
    print '\n9. A vector which is perpendicular to both A and B: ', cross(a,b)
    ab =  np.column_stack([a,b])
    m, n= np.linalg.lstsq(ab,c)[0]
    #print m, n
    print '\n10. linear dependency between A, B, C: %gA %gB = C' % (m, n)
    a_prime = np.matrix('1;2;3')
    b_prime = np.matrix('4;5;6')
    print '\n11. A^TB = ', dot(transpose(a),b), '\nAB^T = \n', dot(a_prime,transpose(b_prime))


    print '##########################################################'
    print '\nThis is problem B:'
    print '\n1. 2A - B = \n', 2 * a2 - b2
    print '\n2. AB = \n', dot(a2, b2), '\nBA = \n', dot(b2, a2)
    print '\n3. (AB)^T = \n', transpose(dot(a2, b2)), '\nB^TA^T = \n', dot(transpose(b2), transpose(a2))
    print '\n4. |A| = \n', np.round(np.linalg.det(a2)), '\n|C| = \n', np.round(
        np.linalg.det(c2))  # I used round fuction in numpy to correct python floating number error
    product_a = dot(a2,transpose(a2))
    product_b = dot(b2,transpose(b2))
    product_c = dot(c2,transpose(c2))
    np.fill_diagonal(product_a,0)
    np.fill_diagonal(product_b,0)
    np.fill_diagonal(product_c,0)
    print '\n 5. The row vectors form an orthogonal set:'
    print '\nA is ', product_a.any() == 0
    print '\nB is ', product_b.any() == 0
    print '\nC is ', product_c.any() == 0
    print '\n6. A^-1 = \n', inverse(a2), '\nB^-1 = \n', inverse(b2)

    print '##########################################################'
    print '\nThis is problem C'
    print '\n1.The eigenvalues and corresponding eigenvectors of A:\n', eig(a3)
    w, v = eig(a3)
    print '2. The matrix V-1AV where V is composed of the eigenvectors of A:\n', np.round(
        dot(dot(inverse(v), a3), v))  # Rounded up
    print '\n3. The dot product between the eigenvectors of A\n', dot(transpose(v)[0], transpose(transpose(v)[1]))
    w2, v2 = eig(b3)
    print '\n4. The dot product between the eigenvectors of A\n', dot(transpose(v2)[0], transpose(transpose(v2)[1]))
    print '\n5. The eigenvectors of B are orthogonal because of B is a symmetric matrix' # Just print out the answer


def dot(a, b):
    return np.dot(a, b)

def cross(a,b):
    return np.cross(a,b)

def angle(a, b):
    return np.arccos(dot(a, b) / mod(a) / mod(b)) * 360 / 2 / np.pi

def mod(a):
    return np.sqrt((a * a).sum())

def perpendicular_vector(a):
#the cross product of two vectors is perpendicular to both of them, unless they're parallel
#credit goes to https://codereview.stackexchange.com/questions/43928/algorithm-to-get-an-arbitrary-perpendicular-vector
    if a[1] == 0 and a[2] == 0:
        if a[0] == 0:
            raise ValueError('zero vector')
    else:
        return np.cross(a, [0, 1, 0])
    return np.cross(a, [1, 0, 0])


def transpose(a):
    return np.transpose(a)


def inverse(a):
    return np.linalg.inv(a)


def eig(a):
    return np.linalg.eig(a)



if __name__ == '__main__':
    main()