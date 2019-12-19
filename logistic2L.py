import numpy as np
# np.set_printoptions(threshold= np.inf)
from PIL import Image
import glob
import tensorflow as tf
import cv2

# global variables
TOTAL_HORSES_TRAIN = 967
TOTAL_ZEBRAS_TRAIN = 1190
TOTAL_PIXELS_PIC = 65536
TOTAL_HORSES_TEST = 120
TOTAL_ZEBRAS_TEST = 140


def logistic_fun(z):
    return 1 / (1.0 + np.exp(-z))


# cvtColor(Image.open(hors), cv2.COLOR_BGR2GRAY)

def making_data_x_from_train():
    train_horses = glob.glob('horses vs zebras/train/horses/*.*')
    train_zebras = glob.glob('horses vs zebras/train/zebras/*.*')
    data_x1 = np.array([[np.array(cv2.cvtColor(cv2.imread(hors), cv2.COLOR_RGB2GRAY))] for hors in train_horses])
    data_x1 = np.array([np.array(mat1.ravel()) for mat1 in data_x1])
    data_x2 = np.array([[np.array(cv2.cvtColor(cv2.imread(zebr), cv2.COLOR_RGB2GRAY))] for zebr in train_zebras])
    data_x2 = np.array([np.array(mat2.ravel()) for mat2 in data_x2])
    data_x_tr = np.concatenate([data_x1, data_x2])
    data_x_tr = np.array([a / 255 for a in data_x_tr])
    return data_x_tr


def making_data_y():
    horses = np.array([[1] for x in range(TOTAL_HORSES_TRAIN)])  # number of pics in train horses
    zebras = np.array([[0] for y in range(TOTAL_ZEBRAS_TRAIN)])  # number of pics in train zebras
    data_y_in = np.concatenate([horses, zebras])
    return data_y_in

(hidden1_size, hidden2_size) = (100, 50)
features = TOTAL_PIXELS_PIC  # number of pixels in each pic
eps = 1e-12
x = tf.compat.v1.placeholder(tf.float32, [None, features])
y_ = tf.compat.v1.placeholder(tf.float32, [None, 1])
W1 = tf.Variable(tf.random.truncated_normal([features, hidden1_size], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))
z1 = tf.nn.relu(tf.matmul(x, W1)+b1)
W2 = tf.Variable(tf.random.truncated_normal([hidden1_size, hidden2_size], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]))
z2 = tf.nn.relu(tf.matmul(z1,W2)+b2)
W3 = tf.Variable(tf.random.truncated_normal([hidden2_size, 1], stddev=0.1))
b3 = tf.Variable(0.)
z3 = tf.matmul(z2, W3) + b3

y = 1 / (1.0 + tf.exp(-z3))

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
loss1 = -(y_ * tf.math.log(y + eps) + (1 - y_) * tf.math.log(1 - y + eps))
loss = tf.reduce_mean(loss1)
update = tf.compat.v1.train.GradientDescentOptimizer(0.00001).minimize(loss)

data_x = making_data_x_from_train()
data_y = making_data_y()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for i in range(0, 10000):
    sess.run(update, feed_dict={x: data_x, y_: data_y})  # BGD

# pdb.set_trace()


test_horses = glob.glob('horses vs zebras/test/horses/*.*')
test_zebras = glob.glob('horses vs zebras/test/zebras/*.*')
data_test_horses = np.array([[np.array(cv2.cvtColor(cv2.imread(hor), cv2.COLOR_RGB2GRAY))] for hor in test_horses])
data_test_zebras = np.array([[np.array(cv2.cvtColor(cv2.imread(zeb), cv2.COLOR_RGB2GRAY))] for zeb in test_zebras])
data_test_horses_flat = np.array([np.array(mat.ravel()) for mat in data_test_horses])
data_test_zebras_flat = np.array([np.array(mat.ravel()) for mat in data_test_zebras])

print("Prediction of horses", np.average(y.eval(session=sess, feed_dict = 	{x:data_test_horses_flat})))

print("Prediction of zebras", np.average(y.eval(session=sess, feed_dict = 	{x:data_test_zebras_flat})))