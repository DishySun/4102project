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
print("Feeding trainning dataset... [1/4]")
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
print("\nBuilding Aj... [2/4]")
for i in range(0, 10):
    A[i] = np.transpose(A[i])

print("\nCalculating SVDs... [3/4]")
U = []
for j in range(0,10):
    u, s, Vh = linalg.svd(A[j],full_matrices=True)
    U.append(u)
    print("%i%% has completed" % ((j+1) * 100 / 10))

print("\nInitializing test data... [4/4]")
k = 28
testImage = []
testLabel = []
testImage_data.read(16)
testLabel_data.read(8)
for i in range(0, 10000):
    testLabel.append(int(testLabel_data.read(1).encode('hex'), 16))
    temp = []
    for j in range(0, 28*28):
        img_pix = int(testImage_data.read(1).encode('hex'), 16)
        temp.append(img_pix)
    testImage.append(temp)
testLabel_data.close()
testImage_data.close()

print("\nInitializing done. Ready for test.\n")
plt.title("Predicting Handwritting Digit")
plt.show(False)

while True:
    input = raw_input("Enter a number from 1 to 10000 ('q' to quit): ")
    if (input == "q"):
        break
    try:
        i = int(input, base=10)
    except ValueError:
        print("not a number")
        continue
    if i < 1 or i > 10000:
        print("not in range")
        continue
    label = testLabel[i-1]
    z = np.transpose(testImage[i-1])
    uk = U[0][0: , 0:k]
    min_lab = 0;
    min_res = norm((np.identity(28*28) - uk.dot(np.transpose(uk))).dot(z), 2)
    for j in range(1, 10):
        uk = U[j][0: , 0:k]
        res = norm((np.identity(28*28) - uk.dot(np.transpose(uk))).dot(z), 2)
        if res < min_res:
            min_res = res
            min_lab = j
    img = np.reshape(np.asarray(testImage[i-1]),(28,28))
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    print("predicted: %i, actual: %i" % (min_lab, label))
    plt.show(False)
    plt.draw()
