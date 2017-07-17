from keras.layers import *
from keras.activations import *
from keras import backend as K
import numpy as np

class GRUStack(Recurrent):


    def __init__(self, output_dim, nb_slots, hidden_size, **kwargs):
        self.output_dim = output_dim
        self.units = output_dim
        self.nb_slots = nb_slots # Stack is actually infinitely deep
        self.hidden_size = hidden_size
        super(GRUStack, self).__init__(**kwargs)  # Init the super class with **kwargs

    @property
    def output_shape(self):
        shape = list(self.input_shape)
        shape[-1] = self.output_dim  # From (batch_size, timesteps, input_dim) to 
        							 # (batch_size, timesteps, output_dim)
        if not self.return_sequences:
            shape.pop(1)  #(batch_size, input_dim)
        return tuple(shape)

    def reset_states(self, states=None):  # reset_states(self):
        nb_samples = self.input_shape[0]  #(batch_size, timesteps, # input_dim)
        S = K.variable(np.zeros((nb_samples, self.nb_slots, 1)))  # Batch x n x 1
        h = K.variable(np.zeros((nb_samples, self.hidden_size))) # Batch x h
       
        self.states = [S, h]
        # Only S and h need at t-1 values
        # Only these 3 variables need to have initial values i.e.
        # at t = 0-
        # Rest all will use these to form other variables

    def get_initial_state(self, inputs):
        S = K.zeros_like(inputs[:, 0, 0]) # Batch
        S = K.stack([S]*self.nb_slots) # n x Batch
        S = K.stack([S]*1) # 1 x n x Batch
        S = K.permute_dimensions(S, (2,1,0)) # Batch x n x 1

        # Similarly for h
        h = K.zeros_like(inputs[:, 0, 0]) # Batch
        h = K.stack([h]*self.hidden_size) # h x Batch
        h = K.permute_dimensions(h, (1, 0)) # Batch x m

        states = [S, h]
        return states

    def build(self, input_shape):
        self.states = [None, None]
        input_dim = input_shape[-1]

        shape = list(input_shape)
        shape[-1] = self.output_dim  # From (batch_size, timesteps, input_dim) 
        							 # to (batch_size, timesteps, output_dim)
        if not self.return_sequences:
            shape.pop(1)  # (batch_size, output_dim)
        output_shape = tuple(shape)
        output_dim = output_shape[-1]

        nb_slots = self.nb_slots
        hidden_size = self.hidden_size
        self.W_U = Dense(hidden_size, input_dim=input_dim) # input_dim x h
        self.W_R = Dense(hidden_size, input_dim=hidden_size)
        self.W_P = Dense(hidden_size, input_dim=2) #k set to 2 : 2 x h
        self.W_V = Dense(output_dim, input_dim=hidden_size)
        self.W_A = Dense(2, input_dim=hidden_size)
        self.W_D = Dense(1, input_dim=hidden_size)
        
        layers = [self.W_U, self.W_R, self.W_P, self.W_V, self.W_A,
                  self.W_D]
        weights = []

        for l in layers:
            weights += l.trainable_weights
            self.trainable_weights = weights

    def step(self, inputs, states):
        S_tm1 = states[0]  # Batch x n x 1
        h_tm1 = states[1]  # Batch x h
        
        S_sq_tm1 = K.squeeze(S_tm1, 2)  # Batch x n
        S_topk_tm1 = S_sq_tm1[:, 0:2] # Batch x 2

        #-----Memory Usage-----#
        # h is Batch x h
        h = tanh( self.W_U(inputs) + self.W_R(h_tm1) + self.W_P(S_topk_tm1))
        # These are written as shape of within brackets : Shape of
        # layers i.e. inputs : W_U
        # Batch x input_dim : input_dim x h
        # Batch x h : h x h
        # Batch x 2 : 2 x h

        #_________Stack Usage________#

        # Action
        a = self.W_A(h)  # Batch x h : h x 2

        # Changing  TOP of Stack

        # First term
        T1_top = K.sigmoid(self.W_D(h)) #Batch x h : h x 1
        T1_top =  K.batch_dot(K.expand_dims(a[:, 0]), T1_top, axes=1)
        # Batch x 1 :. Batch x 1
        # Batch x 1

        # Second term
        T2_top = K.batch_dot(K.expand_dims(a[:, 1]), K.expand_dims(
            S_sq_tm1[:, 1]), axes=1)
        # Batch x 1 :. Batch x 1
        # Batch x 1

        # Temporary variable for the top of the stack
        temp_tos = T1_top + T2_top # Batch x 1

        # For REST of Stack

        n = S_sq_tm1.shape[1] # Get number of slots

        T1_pos_im1 = S_sq_tm1[:, :n-1] # Batch x (1st n-1)
        a_val = K.repeat_elements(K.expand_dims(a[:,0]), n-1, axis=1)
        T1_rest = merge( [a_val, T1_pos_im1], mode='mul' )

        T2_pos_ip1 = S_sq_tm1[:, 1:] # Batch x (last n-1)
        T2_rest = merge([K.repeat_elements(K.expand_dims(a[:, 1]), n-1, axis=1), T2_pos_ip1], mode='mul')

        temp_ros = T1_rest + T2_rest #Batch x n-1

        #Concat to get full updated stack
        S_sq_tos_ros_updated = K.concatenate([temp_tos, temp_ros])

        # Finally expand dim to form S from S_sq
        S = K.expand_dims(S_sq_tos_ros_updated, axis=-1) #Batch x n x 1

        # Final Output
        y = self.W_V(h) # Batch x h : h x output_dim
       
        return y, [S, h]

    def get_config(self):
        config = {
            'output_dim' : self.output_dim,
            'nb_slots' : self.nb_slots,
            'hidden_size' : self.hidden_size
                  }

        base_config = super(GRUStack, self).get_config()

        complete_config = dict(list(base_config.items()) +
                               list(config.items()))
        return complete_config

from keras.models import Sequential

model = Sequential()
model.add(GRUStack(input_shape=(28, 28), output_dim=20, nb_slots=50, hidden_size=50))
model.add(Dense(10, activation='relu'))
model.add(Activation('softmax'))

### Compile
from keras import optimizers
rmsp = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsp,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Test data is the MNIST data
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

data = mnist.train.images.reshape((-1, 28, 28))
labels = mnist.train.labels

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)

from keras.models import load_model
model.save('GRUStack_model.h5')

model.evaluate(mnist.test.images.reshape((-1, 28, 28)), mnist.test.labels)


