import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

def crop_image(img,tol=0):
    # img is 2D image data
    # tol is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

class LMC(): # Learnable Memory Column
    def __init__(self, label):
        self.support = 0
        self.label = label
        self.samples = np.array([])
        self.means = np.array([])
        self.sds = np.array([])

    def learn_sample(self, sample):
        self.support += 1
        if self.support == 1:
            self.samples = sample
            self.means = sample
        else:
            self.samples = np.vstack((self.samples, sample))
            
            x = self.samples
            self.means = np.mean(x, axis=0)
            self.sds = np.std(x, axis=0)

    def distance_score(self, sample):
        return np.mean(np.abs(self.means - sample))

class Cortex():
    def __init__(self):
        self.columns = []
    
    # Returns a reference to the strongest LMC
    def predict_label(self, sample):
        # compare the sample to all columns in the cortex
        scores = []
        for c in self.columns:
            s = c.distance_score(X[i])
            scores.append((c, s))
        
        # fetch that recognizes the sample best
        if len(scores) != 0:
            pred = min(scores, key=lambda t: t[1])[0]
            return pred
        else:
            return None

    def learn_sample(self, label, sample):
        pred = self.predict_label(sample)
        if pred == None or pred.label != label:
            # if the prediction doesn't fit the true label, make a new column with the correct label
            c = LMC(label)
            c.learn_sample(sample)
            self.columns.append(c)
        else:
            # if the prediction is correct, have the predicting column learn the sample
            pred.learn_sample(sample)


# X_ = []
# count = 0
# for x in X:
#     count += 1
#     print(count)
#     x = crop_image(x)
#     x = resize(x, (20, 20))
#     X_.append(x)
# X = np.array(X_)

# np.save('Xp.npy', X)
# print(X.shape)


X, y = np.load('Xp.npy'), np.load('y.npy')

X = X.reshape((60000, 400))
X = X / 255

cortex = Cortex()

tcount = 0
for i in range(5000):
    tcount += 1
    print(tcount)
    sample, label = X[i], y[i]
    cortex.learn_sample(label, sample)

correct = 0
total = 0
for i in range(50000, 60000):
    pred = cortex.predict_label(X[i])
    
    if pred.label == y[i]:
        correct += 1
    else:
        print('Truth: {}, Pred: {}'.format(y[i], pred.label))
        print()
        # plt.imshow(np.reshape(X[i], (20, 20)))
        # plt.show()

        # plt.imshow(np.reshape(pred.means, (20, 20)))
        # plt.show()

    total += 1


print('{} / {} -> {}'.format(correct, total, (correct / total)))

print('Num Columns:', len(cortex.columns))

for c in cortex.columns:
    print('Label: ', c.label)
    print('Support: ', c.support)

    im = np.reshape(c.means, (20, 20))
    plt.imshow(im)
    plt.show()
    print()

