train_samples = 5000

X, y = np.load('Xp.npy'), np.load('y.npy')
Xf, yf = np.load('Xfp.npy'), np.load('yf.npy')
yf = yf + 10

X = X.reshape((60000, 400))
X = X / 255

Xf = Xf.reshape((70000, 400))
Xf = Xf / 255

cortex = Cortex()

print('Training Digits...')
for i in range(train_samples):
    sample, label = X[i], y[i]
    cortex.learn_sample(label, sample)

print('Training Fashion...')
for i in range(train_samples):
    sample, label = Xf[i], yf[i]
    cortex.learn_sample(label, sample)

print('Testing Digits...')
correct = 0
total = 0
support_list = []
for i in range(50000, 51000):
    pred = cortex.predict_label(X[i])
    
    if pred.label == y[i]:
        correct += 1
    else:
        support_list.append(pred.support)

    total += 1

print('{} / {} -> {}'.format(correct, total, (correct / total)))

print()
print('Testing Fashion...')
correctf = 0
totalf = 0
support_list = []
for i in range(50000, 51000):
    pred = cortex.predict_label(Xf[i])
    
    if pred.label == yf[i]:
        correctf += 1
    else:
        support_list.append(pred.support)

    totalf += 1

print('{} / {} -> {}'.format(correctf, totalf, (correctf / totalf)))

print('Total Columns in Cortex:', len(cortex.columns))