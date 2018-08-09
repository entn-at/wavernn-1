import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import math
from tqdm import tqdm


# Next line is used to generate sine wave
function = np.sin(np.arange(hyper_parameters['length_of_wave'])
                  * 2 * np.pi / hyper_parameters['sample_rate']).astype(np.float32)

# Clipping Extra output, so that it can range between [0, 2^16] for label classication
function = np.clip(function * math.pow(2, 15), -math.pow(2, 15),
                   math.pow(2, 15) - 1).astype(np.int16)+2**15

# Calculating coarse and fine parts of the wave
coarse_part = function//256
fine_part = function % 256

# Input and Output vector of both types respectively
X = {
    "coarse": list(),
    "fine": list()
}
y = {
    "coarse": list(),
    "fine": list()
}

# Converting sine wave data into feature sets as described in the paper
for e in range(1, hyper_parameters['length_of_wave']-1):
    # X has been divided by 256 to normalize the input data
    # while y hasnt been because we will be making y as one hot encoding vector.
    X['coarse'].append([coarse_part[e-1]/256.0, coarse_part[e-1]/256.0])
    X['fine'].append(
        [coarse_part[e-1]/256.0, fine_part[e-1]/256.0, coarse_part[e]/256.0])
    y['coarse'].append([coarse_part[e]])
    y['fine'].append([fine_part[e]])


# In[ ]:


# plot of the input function for illustrative purposes
plt.plot(function)


# In[ ]:


# In the same way, coarse has been plotted to better understand the data
plt.plot(X['coarse'][30000:31000])
plt.show()


# In[ ]:


# In the same way, fine has been plotted to better understand the data
# green lines belong to c(t), while orange to f(t)
plt.plot(X['fine'][30000:31000])
plt.show()


# ### This block contains some manipulation over tensors to ease the architecture defined in the next block

# In[ ]:


def get_output_from_deep(type_of_input):
    # In the next line, we calculate index of an element whose value is maximum
    probablistic_best_level = tf.nn.top_k(
        tf.nn.softmax(deep_layer[type_of_input]), 1)[1]
    # Used to normalize the output from the neural netwoks
    return tf.divide(probablistic_best_level, tf.constant(256))


def get_deep_layer(type_of_input):
    # Next line is used to change the shape of the output function, both of these lines denotes a fully connected neural network
    hidden_output = tf.nn.relu(tf.matmul(
        last_time_instant[type_of_input], weights['deep_layer'][type_of_input]) + bias['deep_layer'][type_of_input])
    return tf.matmul(hidden_output, weights['output_layer'][type_of_input]) + bias['output_layer'][type_of_input]
