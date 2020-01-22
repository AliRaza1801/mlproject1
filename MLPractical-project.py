# HandWriting detection
# Detecting digits from images
# where x is the image and y is the ans of img i.e digit identification
# in this project if jpg is of (8 x 8) then you have to convert it in (1 x 64)
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn import datasets,svm

# the digits dataset
digits=datasets.load_digits()

#
print()

image_and_labels=list(zip(digits.images,digits.target))

print()

for index,[image,label] in enumerate(image_and_labels[:5]):
    print()
    plt.subplot(2,5,index+1)#Position num
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    # this interpolation is for clear picturing this is bydefault in new version of python
    plt.title('Training: %i '%label)

# plt.show()

#
#
n_samples=len(digits.images)
print()

imageData=digits.images.reshape((n_samples,-1))
#here double bracket means tuple
print()

#
classifier=svm.SVC(gamma=0.001)
#gamma is learning rate this should be very small

#
classifier.fit(imageData[:n_samples//2],digits.target[:n_samples//2])

#
originalY=digits.target[n_samples//2:]
predictedY=classifier.predict(imageData[n_samples//2:])

image_and_predictions=list(zip(digits.images[n_samples//2:],predictedY))

for index,[image,prediction] in enumerate(image_and_predictions[:5]):
    print()
    plt.subplot(2,5,index+6)#Position num
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    # this interpolation is for clear picturing this is by default in new version of python
    plt.title('Prediction: %i '%prediction)

print("Original Values:",digits.target[n_samples//2:(n_samples//2)+5])
# plt.show()

# I
from scipy.misc import imread,imresize,bytescale

img=imread("IITK ML Data/sam.png")
# or IITK ML Data/FourRB.jpeg
img=imresize(img,(8,8))
classifier =svm.SVC(gamma=0.001)
classifier.fit(imageData[:],digits.target[:])

img=img.astype(digits.images.dtype)
img=bytescale(img,high=16.0,low=0)

print("img.shape: ",img.shape)
print("\n",img)

x_testData=[]
for row in img:
    for col in row:
        x_testData.append(sum(col)/3.0)

print("x_testData: \n",x_testData)

print("len(X_testData):-",len(x_testData))

x_testData=[x_testData]
print("len(x_testData):-",len(x_testData))

result=classifier.predict(x_testData)
print("Machine Output=",result)

plt.show()