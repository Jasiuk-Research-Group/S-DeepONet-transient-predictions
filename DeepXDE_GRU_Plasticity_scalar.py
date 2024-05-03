import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from deepxde.backend import tf
#tf.config.optimizer.set_jit(False)
import os
import deepxde as dde
print(dde.__version__)
dde.config.disable_xla_jit()
from deepxde.data.data import Data
from deepxde.data.sampler import BatchSampler
import keras.backend as K
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


m = 101 # Number of points in the input load amplitude
N_output_frame = 40
HIDDEN = 32
batch_size = 64
fraction_train = 0.8
N_epoch = 300000
N_data = 4000
use_existing_index = False
sub = '_minmax_HD32_PEEQ'

print('\n\nModel parameters:')
print( sub )
print( 'batch_size  ' , batch_size )
print( 'HIDDEN  ' , HIDDEN )
print( 'N_output_frame  ' , N_output_frame )
print( 'm  ' , m )
print( 'fraction_train  ' , fraction_train )
print('\n\n\n')



seed = 123 
tf.keras.backend.clear_session()
try:
    tf.keras.utils.set_random_seed(seed)
except:
    pass
dde.config.set_default_float("float64")



#######################################################################################################################
# Define model
class DeepONetCartesianProd(dde.maps.NN):
    """Deep operator network for dataset in the format of Cartesian product.

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net should be
            equal.
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            activation_branch = activation["branch"]
            self.activation_trunk = dde.maps.activations.get(activation["trunk"])
        else:
            activation_branch = self.activation_trunk = dde.maps.activations.get(activation)

        # User-defined network
        self.branch = layer_sizes_branch[1]
        self.trunk = layer_sizes_trunk[0]
        # self.b = tf.Variable(tf.zeros(1),dtype=np.float64)
        self.b = tf.Variable(tf.zeros(1, dtype=dde.config.real(tf)))

    def call(self, inputs, training=False):
        x_func = inputs[0]
        x_loc = inputs[1]

        # Branch net to encode the input function
        x_func = self.branch(x_func) # [ bs , HD , N_TS ]
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc)) # [ N_pts , HD ]
        # Dot product
        x = tf.einsum("bht,nh->btn", x_func, x_loc)
        # Add bias
        x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
        # return tf.nn.relu( x )


branch = tf.keras.models.Sequential([
     tf.keras.layers.GRU(units=256,batch_input_shape=(batch_size, m, 1),activation = 'tanh',return_sequences = True, dropout=0.00, recurrent_dropout=0.00),
     tf.keras.layers.GRU(units=128,activation = 'tanh',return_sequences = False, dropout=0.00, recurrent_dropout=0.00),
     tf.keras.layers.RepeatVector(HIDDEN),
     tf.keras.layers.GRU(units=128,activation = 'tanh',return_sequences = True, dropout=0.00, recurrent_dropout=0.00),
     tf.keras.layers.GRU(units=256,activation='tanh',return_sequences = True, dropout=0.00, recurrent_dropout=0.00),
     tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(N_output_frame))])
branch.summary()

my_act = "relu"
trunk = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(2,)),
        tf.keras.layers.Dense(101, activation=my_act, kernel_initializer='GlorotNormal'),
        tf.keras.layers.Dense(101, activation=my_act, kernel_initializer='GlorotNormal'), 
        tf.keras.layers.Dense(101, activation=my_act, kernel_initializer='GlorotNormal'),
        tf.keras.layers.Dense(101, activation=my_act, kernel_initializer='GlorotNormal'),
        tf.keras.layers.Dense(101, activation=my_act, kernel_initializer='GlorotNormal'),
        tf.keras.layers.Dense( HIDDEN, activation=my_act , kernel_initializer='GlorotNormal'),
                              ])
trunk.summary()


net = DeepONetCartesianProd(
        [m, branch], [trunk], my_act, "Glorot normal")





#######################################################################################################################
# Load data
base = './Data'
Train_and_test_Amp = np.load(base+'/Amp.npz')['a'].astype(np.float64)[:N_data] # (9903,101)


if 'PEEQ' in sub:
    # PEEQ
    Stress = np.load(base+'/PEEQ.npz')['a'].astype(np.float64)[:N_data] # (9903, 40, 3060)
    smax = 0.6 * np.max(Stress)
else:
    Stress = np.load(base+'/Stress.npz')['a'].astype(np.float64)[:N_data] # (9903, 40, 3060)
    smax = 260.


xy_train_testing = np.load(base+'/Coords.npy').astype(np.float64)[:N_data] # (3060, 2)


# Cap and scale
flag = Stress > smax
print( 'Capped ' , np.sum(flag) / float(len(flag.flatten())) * 100 , ' percent stress data points'  )
Stress[ flag ] = smax


Sshape = Stress.shape
Stress = Stress.reshape([Sshape[0]*Sshape[1],Sshape[2]])
# Scale
scaler = MinMaxScaler()
scaler.fit(Stress)
Stress = scaler.transform(Stress).reshape(Sshape)


N_valid_case = len(Stress)
N_train = int( N_valid_case * fraction_train )

if use_existing_index:
    train_case = np.load('TrainIndex.npy')
else:
    train_case = np.random.choice( N_valid_case , N_train , replace=False )
    np.save('TrainIndex.npy',train_case)

test_case = np.setdiff1d( np.arange(N_valid_case) , train_case )
print('Training with ' , N_train , ' points')


u0_train = Train_and_test_Amp[train_case]
u0_testing =  Train_and_test_Amp[test_case]
s_train = Stress[train_case]
s_testing = Stress[test_case]



# ###################################################################################
# s0_plot = s_train.flatten()
# s1_plot = s_testing.flatten()
# plt.hist( s0_plot , bins=50 , color='r' , alpha=0.6 , density=True ) 
# plt.hist( s1_plot , bins=50 , color='b' , alpha=0.6 , density=True ) 
# plt.legend(['Training' , 'Testing'])
# plt.savefig('train_test_stress_dist.pdf')
# plt.close()
# ###################################################################################


print('u0_train.shape = ',u0_train.shape)
print('u0_testing.shape = ',u0_testing.shape)
print('s_train.shape = ',s_train.shape)
print('s_testing.shape = ',s_testing.shape)
print('xy_train_testing.shape', xy_train_testing.shape)

x_train = (u0_train, xy_train_testing)
y_train = s_train 
x_test = (u0_testing, xy_train_testing)
y_test = s_testing


class TripleCartesianProd(Data):
    """Dataset with each data point as a triple. The ordered pair of the first two
    elements are created from a Cartesian product of the first two lists. If we compute
    the Cartesian product of the first two arrays, then we have a ``Triple`` dataset.

    This dataset can be used with the network ``DeepONetCartesianProd`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays. The first element has the shape (`N1`,
            `dim1`), and the second element has the shape (`N2`, `dim2`).
        y_train: A NumPy array of shape (`N1`, `N2`).
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

        self.branch_sampler = BatchSampler(len(X_train[0]), shuffle=True)
        self.trunk_sampler = BatchSampler(len(X_train[1]), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        if not isinstance(batch_size, (tuple, list)):
            indices = self.branch_sampler.get_next(batch_size)
            return (self.train_x[0][indices], self.train_x[1]), self.train_y[indices]
        indices_branch = self.branch_sampler.get_next(batch_size[0])
        indices_trunk = self.trunk_sampler.get_next(batch_size[1])
        return (
            self.train_x[0][indices_branch],
            self.train_x[1][indices_trunk],
        ), self.train_y[indices_branch, indices_trunk]

    def test(self):
        return self.test_x, self.test_y



data = TripleCartesianProd(x_train, y_train, x_test, y_test)



#######################################################################################################################
def err( y_train , y_pred ):
    ax = -1
    return np.linalg.norm( y_train - y_pred , axis=ax ) / ( np.linalg.norm( y_train , axis=ax ) + 1e-8 )

def inv( vec , scaler ):
    my_shape = vec.shape
    return scaler.inverse_transform(vec.reshape([my_shape[0]*my_shape[1],my_shape[2]])).reshape(my_shape)

def L2( y_train , y_pred ):
    y_train_original = inv( y_train , scaler )
    y_pred_original = inv( y_pred , scaler )
    return np.mean( err( y_train_original , y_pred_original ).flatten() )

def ABS( y_train , y_pred ):
    y_train_original = inv( y_train , scaler )
    y_pred_original = inv( y_pred , scaler )
    return np.mean( np.abs( y_train_original - y_pred_original ).flatten() )

# def MSE_ori( y_train, y_pred ):
#     y_train_original = inv( y_train , scaler )
#     y_pred_original = inv( y_pred , scaler )
#     tmp = tf.math.square( K.flatten(y_train_original) - K.flatten(y_pred_original) )
#     return tf.math.reduce_mean(tmp)

def MSE( y_true, y_pred ):
    tmp = tf.math.square( K.flatten(y_true) - K.flatten(y_pred) )
    return tf.math.reduce_mean(tmp)



# Build model
model = dde.Model(data, net)
# Stage 1 training
model.compile(
    "adam",
    lr=5e-4,
    decay=("inverse time", 1, 1e-4),
    loss=MSE,
    metrics=[ L2 , ABS ],
)
losshistory1, train_state1 = model.train(iterations=N_epoch, batch_size=batch_size, model_save_path="./mdls/TrainedModel"+sub)
np.save('losshistory'+sub+'.npy',losshistory1)


import time as TT
st = TT.time()
y_pred = model.predict(data.test_x)
duration = TT.time() - st
print('y_pred.shape =', y_pred.shape)
print('Prediction took ' , duration , ' s' )
print('Prediction speed = ' , duration / float(len(y_pred)) , ' s/case' )

y_test_original = inv( y_test , scaler )
y_pred_original = inv( y_pred , scaler )
np.savez_compressed('TestData'+sub+'.npz',a=y_test_original,b=y_pred_original,c=u0_testing,d=xy_train_testing)


error_s = err( y_test_original , y_pred_original )
error_s[ error_s > 1. ] = 1.

# print("error_s = ", error_s)
np.save( 'errors'+sub+'.npy' , error_s )

print('mean of relative L2 error of s: {:.2e}'.format( np.mean(error_s) ))
print('std of relative L2 error of s: {:.2e}'.format( np.std(error_s) ))

# plt.hist( error_s.flatten() , bins=25 )
# plt.savefig('Err_hist_DeepONet'+sub+'.jpg' , dpi=1000)
