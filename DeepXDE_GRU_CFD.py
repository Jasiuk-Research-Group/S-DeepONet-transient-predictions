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


m = 100 # Number of points in the input load amplitude
N_output_frame = 25
out_frame_idx = np.arange(100)[::4] # Every 4 frames
HIDDEN = 32
batch_size = 64
fraction_train = 0.8
N_data = 4000
use_existing_index = False
out_list = ['P','U','V']
sub = '_cfd_PUV_4000_ori_HD32'

N_component = len(out_list)
print('\n\nModel parameters:')
print( sub )
print( 'batch_size  ' , batch_size )
print( 'HIDDEN  ' , HIDDEN )
print( 'N_component  ' , N_component )
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

        # print( x_func.shape )
        # print( x_loc.shape )
        # exit()

        # Branch net to encode the input function
        x_func = self.branch(x_func) # [ bs , HD , N_TS ]
        # Trunk net to encode the domain of the output function
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x_loc = self.activation_trunk(self.trunk(x_loc)) # [ N_pts , HD , N_comp ]

        # Dot product
        x = tf.einsum("bht,nhc->btnc", x_func, x_loc)
        
        # Add bias
        x += self.b

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x


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
        tf.keras.layers.Dense( HIDDEN * N_component , activation=my_act , kernel_initializer='GlorotNormal'),
        tf.keras.layers.Reshape( [ HIDDEN , N_component ] ),
                              ])
trunk.summary()

net = DeepONetCartesianProd(
        [m, branch], [trunk], my_act, "Glorot normal")





#######################################################################################################################
# Load data
Train_and_test_Amp = np.load('../all_Amp.npz')['a'].astype(np.float64)
xy_train_testing = np.load('../all_Coord.npz')['a'].astype(np.float64)

# ###################################################################################
# # Scale check
# for fn in ['P','U','V','O']:
#     data = np.load('all_'+fn+'.npz')['a'][ : , out_frame_idx , : ].flatten()
#     plt.figure()
#     plt.hist( data , bins=100 , color='r' , alpha=0.6 , density=True ) 
#     plt.title( fn )
#     plt.savefig( fn+'_distribution.pdf')
#     plt.close()
# exit()
# ###################################################################################


arry = []
scalers = []
Data_caps = { 'P': 3.,
              'U': 1.5,
              'V': 0.3,
              'O': 25.}
for fn in out_list:
    print('Output variable ' + fn )
    # data = np.load('all_'+fn+'.npz')['a'][ : , out_frame_idx , : ].astype(np.float64)

    data = np.load('../4000_'+fn+'.npz')['a']
    data = data.astype(np.float64)

    # Cap and scale
    smax = Data_caps[ fn ]
    flag = data > smax
    print( 'Capped ' , np.sum(flag) / float(len(flag.flatten())) * 100 , ' percent stress data points, positive'  )
    data[ flag ] = smax

    flag = data < -smax
    print( 'Capped ' , np.sum(flag) / float(len(flag.flatten())) * 100 , ' percent stress data points, negative'  )
    data[ flag ] = -smax

    # data /= smax

    # Time-step dependent scaling
    tmp = np.max( data , axis=-1 ) # max over domain
    time_dep_scale = np.max( tmp , axis = 0 ) # Max over cases
    scale_inv = 1. / time_dep_scale
    data = np.einsum( 'ijk,j->ijk' , data , scale_inv )

    # ###################################################################################
    # plt.plot( time_dep_scale , 'k' ) 
    # plt.xlabel('Time step')
    # plt.ylabel('Max data scale')
    # plt.savefig('Max_data_scale_'+fn+'.pdf')
    # plt.close()
    # ###################################################################################

    arry.append( data.copy() )
    scalers.append( time_dep_scale.copy() )

# Combine all data
Stress = np.stack(arry, axis=-1)

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
# plt.savefig('train_test_dist.pdf')
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
    return np.einsum( 'ijk,j->ijk' , vec , scaler )


def L2_P( y_train , y_pred ):
    ccc = 0
    y_train_original = inv( y_train[:,:,:,ccc] , scalers[ccc] )
    y_pred_original = inv( y_pred[:,:,:,ccc] , scalers[ccc] )
    return np.mean( err( y_train_original , y_pred_original ).flatten() )

def L2_U( y_train , y_pred ):
    ccc = 1
    y_train_original = inv( y_train[:,:,:,ccc] , scalers[ccc] )
    y_pred_original = inv( y_pred[:,:,:,ccc] , scalers[ccc] )
    return np.mean( err( y_train_original , y_pred_original ).flatten() )

def L2_V( y_train , y_pred ):
    ccc = 2
    y_train_original = inv( y_train[:,:,:,ccc] , scalers[ccc] )
    y_pred_original = inv( y_pred[:,:,:,ccc] , scalers[ccc] )
    return np.mean( err( y_train_original , y_pred_original ).flatten() )

def MSE( y_true, y_pred ):
    tmp = tf.math.square( K.flatten(y_true) - K.flatten(y_pred) )
    return tf.math.reduce_mean(tmp)




# Build model
model = dde.Model(data, net)
# Training
model.compile(
    "adam",
    lr=5e-4,
    decay=("inverse time", 1, 1e-4),
    loss=MSE,
    metrics=[ L2_P , L2_U , L2_V ],
)
losshistory1, train_state1 = model.train(iterations=300000, batch_size=batch_size, model_save_path="./mdls/TrainedModel_"+sub)
np.save('losshistory'+sub+'.npy',losshistory1)


#############################################################################################################################
import time as TT
st = TT.time()
y_pred = model.predict(data.test_x)
duration = TT.time() - st
print('y_pred.shape =', y_pred.shape)
print('Prediction took ' , duration , ' s' )
print('Prediction speed = ' , duration / float(len(y_pred)) , ' s/case' )

gt , pred = [] , []
for ccc in range(3):
    gt.append( inv( y_test[:,:,:,ccc] , scalers[ccc] ).copy() )
    pred.append( inv( y_pred[:,:,:,ccc] , scalers[ccc] ).copy() )

    error_s = err( gt[-1] , pred[-1] )
    error_s[ error_s > 1. ] = 1.
    print('mean of relative L2 error of '+out_list[ccc]+': {:.2e}'.format( np.mean(error_s) ))
    print('std of relative L2 error of '+out_list[ccc]+': {:.2e}'.format( np.std(error_s) ))
    np.save( out_list[ccc]+'_err'+sub+'.npy' , error_s )

np.savez_compressed('TestData'+sub+'_test.npz',a=gt[0],b=gt[1],c=gt[2])
np.savez_compressed('TestData'+sub+'_pred.npz',a=pred[0],b=pred[1],c=pred[2])
np.savez_compressed('TestData'+sub+'_u.npz',a=u0_testing,b=xy_train_testing)