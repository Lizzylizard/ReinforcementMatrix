import numpy as np, tensorflow as tf;
import PIL.Image as Img;
import os, sys;

B = 200;
max_iter = 10000;
eps = 0.0005;
L = 500;
imgW = 24
imgH = 1
imgC = 1;
nrClasses = 4
rowLen = imgW * imgH * imgC;
splitFactor = 0.7;
leaveOut = eval(sys.argv[2]);
oversampling = True;


# rowLen = 5 ;

def loadImgDir(path):
  filelist = os.listdir(path);
  imgObjs = []
  for f in sorted(filelist):
    if f.find(".png") != -1:
      o = Img.open(path + "/" + f);
      o.load();
      imgObjs.append(np.array(o).ravel());
  imgArr = np.array(imgObjs);

  # convert 2 one-hot
  numLabels = np.array(
    [int(f.split("_")[0].replace("class", "")) for f in filelist]);
  oneHot = np.zeros([len(numLabels), 4]);
  oneHot[range(0, len(numLabels)), numLabels] = 1;

  return imgArr, oneHot, numLabels[:, np.newaxis];


infile = open(sys.argv[1], "rb");
X, T = np.load(infile).values();
print(X.shape, T.shape);
X[:, leaveOut] = 0;

Tn = T.argmax(axis=1)
# X,T,Tn = loadImgDir(sys.argv[1]) ;
print(T.sum(axis=0))
# T = Tn # for regression

if oversampling:
  presentClasses = list(range(0, nrClasses));
  classIndices = {};
  numClasses = T.argmax(axis=1);
  maxSamples = -1;
  maxClass = -1;
  for prClass in presentClasses:
    ind = np.where(numClasses == prClass)[0];
    print("indices for class ", prClass, " are: ", len(ind));
    classIndices[prClass] = ind;
    nrSamples = len(ind);
    if nrSamples > maxSamples:
      maxSamples = nrSamples;
      maxClass = prClass;
    pass;
  print("Most frequent class: ", maxClass, " --> # inst = ",
        maxSamples);

  totalIndices = [];
  for prClass in presentClasses:
    ind = classIndices[prClass];
    if len(ind) == 0: continue;
    selectIndices = np.random.randint(0, len(ind), size=[maxSamples]);
    resampledIndices = ind[selectIndices];
    print("Resampled for class ", prClass, resampledIndices.shape);
    totalIndices.append(resampledIndices);
  allIndices = np.concatenate(totalIndices);
  print(allIndices.shape);

  X = X[allIndices];
  T = T[allIndices];
  Tn = Tn[allIndices];

  print("Final distro; ", T.sum(axis=0));

if True:  # shuffle train data
  indices = np.arange(0, X.shape[0]);
  np.random.shuffle(indices);
  X = X[indices, :];
  T = T[indices, :]
  Tn = Tn[indices]

if rowLen < imgH * imgW * imgC:
  print("!!!!!!!!!!!!!!!!PCA!!!!!!!!!!!!!!!!!!!");
  X_ = X[0:10000];
  C = (X_[:, :, np.newaxis] * X_[:, np.newaxis, :]).mean(axis=0);
  print(C.shape);
  eigvals, eigvecs = np.linalg.eigh(C);
  Xprime = np.dot(X, eigvecs);
  print(Xprime.shape, eigvecs.shape);
  newX = Xprime[:, 0:rowLen];
  X = newX;

# train test split
nu = splitFactor;
splitIndex = int(X.shape[0] * nu);
traind = X[0:splitIndex, :];
trainl = T[0:splitIndex, :];
testd = X[splitIndex:, :];
testl = T[splitIndex:, :]

# Definition of model
input = tf.placeholder(tf.float64, [None, rowLen]);
targets = tf.placeholder(tf.float64, [None, nrClasses]);

a0 = tf.reshape(input, (-1, rowLen));

W1 = tf.Variable(np.random.uniform(-0.01, 0.01, [rowLen, L]),
                 name="W1");
b1 = tf.Variable(np.random.uniform(-0.01, 0.01, [L]), name="b1");
a1 = tf.nn.relu_layer(a0, W1, b1, "a1");

W2 = tf.Variable(np.random.uniform(-0.01, 0.01, [L, L]), name="W2");
b2 = tf.Variable(np.random.uniform(-0.01, 0.01, [L]), name="b2");
a2 = tf.nn.relu_layer(a1, W2, b2, "a2");

W3 = tf.Variable(np.random.uniform(-0.01, 0.01, [L, L]), name="W3");
b3 = tf.Variable(np.random.uniform(-0.01, 0.01, [L]), name="b3");
a3 = tf.nn.relu_layer(a2, W3, b3, "a3");

W4 = tf.Variable(np.random.uniform(-0.01, 0.01, [L, nrClasses]),
                 name="W4");
b4 = tf.Variable(np.random.uniform(-0.01, 0.01, [1, nrClasses]),
                 name="b4");
a4 = tf.matmul(a3, W4) + b4;

loss = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits_v2(targets,
                                             a4));  # for classification
# loss = tf.losses.mean_squared_error(targets,a4)  # for regressopm

# classification error measurement
comp = tf.equal(tf.argmax(targets, axis=1), tf.argmax(a4, axis=1));
ce = 1. - (tf.reduce_mean(tf.cast(comp, tf.float64)));
# ce = loss ; # regression


# optimization: by executing updateOp, we perform a single iteration of training (see below)
# sgdObj = tf.train.GradientDescentOptimizer(learning_rate=eps) ;
sgdObj = tf.train.AdamOptimizer(learning_rate=eps);
updateOp = sgdObj.minimize(loss);

sess = tf.Session();
sess.run(tf.global_variables_initializer());

nrBatches = traind.shape[0] // B;

print("Batch size = ", B, ", nrBatches=", nrBatches);
batchIndex = 0;
for it in range(0, max_iter):
  dataBatch = traind[batchIndex * B:(batchIndex + 1) * B, :];
  targetBatch = trainl[batchIndex * B:(batchIndex + 1) * B, :];

  sess.run(updateOp,
           feed_dict={input: dataBatch, targets: targetBatch});
  if it % 30 == 0:
    testErr = sess.run(ce, feed_dict={input: testd, targets: testl});
    print(it, "Test error is ", testErr);
  batchIndex += 1;
  if batchIndex >= nrBatches:
    batchIndex = 0;

# CM calc
outputs = sess.run(a4, feed_dict={input: testd, targets: testl});
numOutputs = outputs.argmax(axis=1);
cm = np.zeros([nrClasses, nrClasses]);
for numOut, l in zip(numOutputs, testl.argmax(axis=1)):
  cm[l, numOut] += 1;

print("Confusion Matrix");
print(cm);
