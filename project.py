import numpy as np
from scipy import linalg
from scipy.linalg import norm
from matplotlib import pyplot as plt

#load trainning files
trainImage_path = "train-images-idx3-ubyte"
try:
    trainImage_data = open(trainImage_path, mode='r')
except IOError:
    print("Fail to load 'trainning image' file '%s'" % trainImage_path)
    exit()

trainLabel_path = "train-labels-idx1-ubyte"
try:
    trainLabel_data = open(trainLabel_path, mode='r')
except IOError:
    print("Fail to load 'trainning label' file '%s'" % trainLabel_path)
    exit()

testImage_path = "t10k-images-idx3-ubyte"
try:
    testImage_data = open(testImage_path, mode='r')
except IOError:
    print("Fail to load 'test image' file '%s'" % testImage_path)
    exit()

testLabel_path = "t10k-labels-idx1-ubyte"
try:
    testLabel_data = open(testLabel_path, mode='r')
except IOError:
    print("Fail to load 'test lable' file '%s'" % testLabel_path)
    exit()


#meta data
#magic number, usless and drop
trainImage_data.read(4)
#number of images = 60,000
TRAIN_IMAGE_NUMBER = int(trainImage_data.read(4).encode('hex'), 16)
#number of rows = 28
ROW_NUMBER = int(trainImage_data.read(4).encode('hex'), 16)
#number of columns = 28
COLUMN_NUMBER = int(trainImage_data.read(4).encode('hex'), 16)
#pixel number for each image = 784
PIXEL_NUMBER = ROW_NUMBER * COLUMN_NUMBER
#skip non-label data
trainLabel_data.read(8)

#trainning
print("Feeding trainning dataset...")
A = [[], [], [], [], [], [], [], [], [], []]
for i in range(0, TRAIN_IMAGE_NUMBER):
    lab = int(trainLabel_data.read(1).encode('hex'), 16)
    temp = []
    for j in range(0, PIXEL_NUMBER):
        img_pix = int(trainImage_data.read(1).encode('hex'), 16)
        temp.append(img_pix)
    A[lab].append(temp)
    if (i == 15000 or i == 30000 or i == 45000):
        print("%i%% has completed" % (100 * i / TRAIN_IMAGE_NUMBER))
trainImage_data.close()
trainLabel_data.close()
print("done")
print("\nChecking feeding process...")
for i in range(0, 10):
    print("A[%i] size: %i" % (i, len(A[i])) )
    print("A[%i][100] size: %i" % (i, len(A[i][100])))
print("\nBuilding Aj...")
for i in range(0, 10):
    A[i] = np.transpose(A[i])
print("\nChechking build Aj process...")
for i in range(0,10):
    m , n = A[i].shape
    print("A[%i] : %i x %i" % (i, m , n))

print("\nCalculating SVDs..")
U = []
for j in range(0,10):
    u, s, Vh = linalg.svd(A[j],full_matrices=True)
    U.append(u)
    print("%i%% has completed" % ((j+1) * 100 / 10))

print("\nTesting...")
K = np.arange(1,52)
correct = np.zeros(51, dtype=int)
#magic number, usless and drop
testImage_data.read(4)
#number of images = 10,000
TEST_IMAGE_NUMBER = int(testImage_data.read(4).encode('hex'), 16)
#skip non-image data
testImage_data.read(8)
#skip non-label data
testLabel_data.read(8)
for i in range(0,TEST_IMAGE_NUMBER):
    lab = int(testLabel_data.read(1).encode('hex'), 16)
    temp = []
    for j in range(0, PIXEL_NUMBER):
        img_pix = int(testImage_data.read(1).encode('hex'), 16)
        temp.append(img_pix)
    z = np.transpose(temp)
    for k in K:
        uk = U[0][0: , 0:k]
        min_lab = 0;
        min_res = norm((np.identity(PIXEL_NUMBER) - uk.dot(np.transpose(uk))).dot(z), 2)
        for j in range(1, 10):
            uk = U[j][0: , 0:k]
            res = norm((np.identity(PIXEL_NUMBER) - uk.dot(np.transpose(uk))).dot(z), 2)
            if res < min_res:
                min_res = res
                min_lab = j
        if min_lab == lab:
            correct[k-1] += 1
    print("%i/10000 has completed" % (i+1))

testLabel_data.close()
testImage_data.close()
acc = correct * 100.0 / TEST_IMAGE_NUMBER
print("Ploting...")
plt.title("Handwritting Digit")
plt.xlabel("k")
plt.ylabel("accuracy(%)")
plt.plot(K,acc,color='b')
plt.plot(K,acc,'ob',color='r')
plt.axis([0,52,80, 96])
plt.show()
