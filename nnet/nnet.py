
import scipy.io.arff as arff
import numpy as np
from collections import OrderedDict
from collections import Counter
import sys


"""
Function to Parse Arff File and Save Data
input: name (string) of file
output: (tuple)
  1) data (dictionary), where keys are column names
  2) metadata (obj) on the attributes
"""
def parse_arff(name):

  # extract using arff package
  file = arff.loadarff(open(name, 'rb'))
  raw_data, metadata = file
  data = [[v if type(v) is np.string_ else round(v, 14) for v in l] for l in raw_data]
  return data, metadata

"""
Standardize normal features
"""
def stand(m, train, test):

  # loop through the features standardize numeric ones
  feature_types = [m[n][0] for n in train_m.names()]
  for i in range(len(feature_types)):
    t = feature_types[i]
    if t == "numeric":
      # collect training data
      data = []
      for n in range(len(train)):
        data.append( train[n][i] )
      m = np.mean(data)
      sd = np.std(data)

      # standardize training data
      for n in range(len(train)):
        train[n][i] = (train[n][i] - m) / sd

      # standardize testing data
      for n in range(len(test)):
        test[n][i] = (test[n][i] - m) / sd

  # standardized data
  return train, test

"""
Sigmoid function
"""
def sigmoid(x):
  return 1 / float(1 + np.exp(-x))

"""
Randomize weights b/n -.01, .01
"""
def random():
  return np.random.uniform(-0.01, 0.01)

"""
Node class for internal nodes
"""
class net:

  """
  Initialize net, randomize weights
  object has following vars:
  - learning
  - h
  - response
  - net: input weights, hidden weights
  """
  def __init__(self, m, h, n):

    # save learning rate & hidden units
    self.learning = n
    self.h = h

    # save features
    self.features = m.names()
    self.features.pop()

    # save responses
    response = m['class'][1]
    self.response = {response[0]: 0, response[1]: 1}

    # initialize input weights
    input_layer = []
    for i in range(self.h+1):

      input_weights = {}

      # initialize the structure for the net
      for n in self.features:
        feature_info = m[n]
        feature_type = feature_info[0]
        if feature_type == 'numeric':
          input_weights[n] = {'type': feature_type, 'weights': random()}
        else:
          feature_values = {f:random() for f in feature_info[1]}
          input_weights[n] = {'type': feature_type, 'weights': feature_values}

      # add to a list
      input_layer.append(input_weights)

    # remove extra if h > 0
    if self.h > 0:
      input_layer.pop()

    # initialize hidden units
    hidden_units = np.array([random() for i in range(self.h)])

    # populate network
    self.net = {'input': input_layer, 'hidden': hidden_units}

  """
  run a training instance through the neural network and run backprop
  """
  def compute_output(self, instance, train):

    # find y by taking the last value if instance
    y = self.response[list(instance).pop()]

    # run data through each feature, accounting for h
    input_layer_out = []
    for l in self.net['input']:
      input_outputs = []
      for i in range(len(self.features)):
        x_i = instance[i]

        # extract
        f = l[ self.features[i] ]
        if f['type'] == "numeric":
          input_outputs.append( f['weights']*x_i )
        else:
          input_outputs.append( f['weights'][x_i] )

      # calculate values needed to move on
      input_layer_out.append( sigmoid(sum(input_outputs)) )

    # finalize input layer weights
    input_layer_out = np.array( input_layer_out )

    # run through hidden units - use sigmoid function on output units
    if self.h > 0:
      final_out = sigmoid( sum(self.net['hidden'] * input_layer_out) )

    # don't need to run through hidden units
    else:
      final_out = input_layer_out[0]

    # calculate errors
    CE_error = -y * np.log(final_out) - (1 - y) * np.log(1 - final_out)
    accurate = y == round(final_out)

    # backprop
    if train:
      self.backprop(instance, input_layer_out, final_out, y)
      return CE_error, accurate
    else:
      return final_out, y

  """
  backprop algorithm: updates the weights
  """
  def backprop(self, instance, input_outputs, p, y):

    # delta for output layer (back to hidden layer)
    output_delta = y - p

    # find delta at hidden unit (if it exist) & update weights
    if self.h > 0:

      # delta for the hidden layer (back to input layer)
      hidden_delta = input_outputs * (1 - input_outputs) * output_delta * self.net['hidden']

      # adjust weights for hidden layer
      change_weight_hidden = self.learning * output_delta * input_outputs
      self.net['hidden'] += change_weight_hidden
      next_delta = hidden_delta
    else:
      next_delta = [output_delta]

    # adjust weights for each feed into hidden layer
    for j in range(len(self.net['input'])):
      l = self.net['input'][j]

      # update weights at input stage
      for i in range(len(self.features)):
        x_i = instance[i]
        f = l[ self.features[i] ]
        if f['type'] == "numeric":
          f['weights'] += self.learning * next_delta[j] * x_i
        else:
          f['weights'][x_i] += self.learning * next_delta[j] * 1

  """
  train neural net
  """
  def train_net_epoch(self, d, epoch):

    # randomize training order
    order = np.random.permutation(len(d))

    # train network
    CE_error = []
    accurate = []
    for i in order:
      error, correct = self.compute_output(d[i], True)
      CE_error.append(error)
      accurate.append(correct)

    # output results
    CE_error = sum(CE_error)
    correct_class = sum(accurate)
    incorrect_class = len(accurate) - sum(accurate)
    print '%s\t%.05f\t%s\t%s' % (epoch, CE_error, correct_class, incorrect_class)

  """
  test neural net
  """
  def test_net(self, d):

    # loop through and test on data (no backprop)
    accurate = []
    for i in range(len(d)):
      p, y = self.compute_output(d[i], False)
      print '%.5f\t%s\t%s' % (p, [k for k, v in self.response.items() if v == round(p)][0], [k for k, v in self.response.items() if v == y][0])
      accurate.append(y == round(p))
    print '%s\t%s' % (sum(accurate), len(accurate) - sum(accurate))



if __name__ == '__main__':

  args = sys.argv

  # program arguments
  n = float(args[1])
  h = int(args[2])
  e = int(args[3])

  # train/test files
  train_file = args[4]
  test_file = args[5]

  # open data
  train_d, train_m = parse_arff(train_file)
  test_d, test_m = parse_arff(test_file)

  train_d, test_d = stand(train_m, train_d, test_d)

  # initialize
  n = net(train_m, h, n)
  # train
  for i in range(e):
    n.train_net_epoch(train_d, i+1)
  # test
  n.test_net(test_d)
