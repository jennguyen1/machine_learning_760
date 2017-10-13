
import scipy.io.arff as arff
import numpy as np
from collections import Counter
import sys

"""
Node class for internal nodes
"""
class node:

  """
  Initialize class
    node_type: 'internal' or 'leaf'
    node_name: signify the split class
    data: all data corresponding to node_name
    metadata: original metadata
    feature_to_split: best feature to use for splitting
    splits: datasets spilt by best feature
    responses: responses of leaf
  """
  def __init__(self, node_type, node_name, data, metadata, feature_to_split = None, splits = None, responses = None):

    # process response counts
    response_name = metadata.names()[-1]
    response_types = metadata[response_name][1]
    data_counts = Counter(data[response_name])
    self.response_counts = [data_counts[response_types[0]], data_counts[response_types[1]]]

    # defined node attributes
    self.node_type = node_type
    self.node_name = node_name
    self.fix = []

    # define printing order, given by the node_name's type
    if node_type is 'internal':
      self.print_order = metadata[feature_to_split][1]

    # internal node attributes
    self.feature = feature_to_split
    self.splits = splits
    if self.node_type is 'internal':
      self.children = {}
      for k in splits:
        self.children[k] = None

    # leaf nodes attributes
    elif self.node_type is 'leaf':
      self.__tally_results(responses)

  """
  String representation of node
  """
  def __repr__(self):

    s = "%s %s" % (self.node_name, str(self.response_counts))
    if self.node_type is 'leaf':
      s += ': %s' % self.predictions
    return s

  """
  Recursive function to print out nodes
  """
  def print_all(self, i):

    if self.node_name is not 'root':
      print '|\t'*i + str(self)

    if self.node_type is 'internal':
      if self.print_order is None:
        number = float(self.children.keys()[0].split()[-1])
        self.children['%s <= %.6f' % (self.feature, number)].print_all(i + 1)
        self.children['%s > %.6f' % (self.feature, number)].print_all(i + 1)
      else:
        for j in self.print_order:
          self.children['%s = %s' % (self.feature, j)].print_all(i + 1)

  """
  Get/set predictions for leaf nodes
  """
  # tallies responses to predict class
  def __tally_results(self, responses):

    # tally up responses
    tally = Counter(responses)
    counts = tally.values()

    # if there is only one value of response left
    if len(tally) == 1:
      self.predictions = tally.keys()[0]

    # if not enough info, need to refer to parent node
    elif len(tally) == 0 or counts[0] == counts[1]:
      self.predictions = None

    # otherwise assign to most common class
    else:
      max_count_index = np.argmax(counts)
      self.predictions = tally.keys()[max_count_index]

  # tallies results if needed from parent class
  def assign_predictions(self, parent_responses):

    if self.predictions is None:
      self.__tally_results(parent_responses)

  def get_predictions(self):
    return self.predictions

  """
  Get/set children nodes for internal nodes
  """
  def add_children(self, key, node):
    if self.node_type is 'internal':
      self.children[key] = node

  def get_children(self):
    if self.node_type is 'internal':
      return self.children

  """
  Get node attributes
  """
  def get_feature(self):
    if self.node_type is 'internal':
      return self.feature

  def get_type(self):
    return self.node_type

  """
  Process data
  """
  def predict(self, observation):

    # find the right child to pass to
    if '<' in self.children.keys()[0] or '>' in self.children.keys()[0]:
      numeric_threshold = float(self.children.keys()[0].split()[-1])
      direction = ['>', '<='][observation <= numeric_threshold]
    else:
      direction = observation

    # find the correct child and return
    use_key = [x for x in self.children.keys() if ' %s' % direction in x][0]
    return self.children[use_key]

  """
  Add fix
  """
  def add_fix(self, x):
    self.fix.append(x)

  def get_fix(self):
    return self.fix

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
  training_data, metadata = file

  # save data into dictionary
  data = {}
  for i in range(len(metadata.names())):
    feature = [l[i] for l in training_data]
    data[metadata.names()[i]] = np.array(feature)

  return data, metadata

"""
Function to Calculate Information Gain of one Feature
inputs:
  1) y (numpy array) of the responses
  2) x (numpy array) of the feature
  3) x_type (string) type of feature (numeric or nominal)
  4) numeric_threshold (number) value to cutoff the split
outputs: (tuple)
  1) information gain (float)
  2) numeric threshold (number/None) for the split if feature is numeric
"""
def calc_info_gain(y, x, x_type, numeric_threshold = None):

  # info y only #
  unique_y = Counter(y)
  response = np.array( unique_y.values() )
  y_prop = response / float(sum(response))
  Hy = - sum(np.multiply(y_prop, np.log2(y_prop)))

  # info for y given x #
  if x_type is 'nominal':

    # use threshold to compare if numeric values given
    if numeric_threshold is not None:
      use_x = x <= numeric_threshold
    else:
      use_x = x

    # count up unique values of feature x
    unique_x = Counter(use_x)
    feature = np.array( unique_x.values() )
    feature_prop = feature / float(sum(feature))

    # info for y given each x
    Hyx = []
    for k in unique_x.keys():
      # find proportion of x in all data
      feature_prop = unique_x[k] / float(sum(feature))

      # info for y given specific x
      index = [p for p, v in enumerate(use_x) if v == k]
      sub_responses = np.array(Counter(y[index]).values())
      sub_y_prop = sub_responses / float(sum(sub_responses))
      Hyx_specific = - sum(np.multiply(sub_y_prop, np.log2(sub_y_prop)))

      # combine prop and Hyx for specifc x
      Hyx.append(Hyx_specific * feature_prop)

  elif x_type is 'numeric':

    # find all unique x
    unique_x = np.sort( list(set(x)) )

    # find the right split threshold
    splits = []
    split_info = []
    for i in range(len(unique_x)-1):

      # extract y for each group
      y_i = y[x == unique_x[i]]
      y_ip1 = y[x == unique_x[i+1]]

      # find differences
      unique_yi = Counter(y_i).keys()
      unique_yip1 = Counter(y_ip1).keys()

      # if the classes between the two don't match or if
      if unique_yi is not unique_yip1 or len(unique_yi) == 2 or len(unique_yip1) == 2:
        splits.append( (unique_x[i] + unique_x[i+1]) / 2 )

    # compute info for potential numeric splits
    for s in splits:
      split_info.append( calc_info_gain(y, x, "nominal", numeric_threshold = s)[0] )

    if len(unique_x) == 1:
      splits.append(unique_x)
      split_info.append( 0 )

  # information gain (argmax takes the first instance that is the max)
  if x_type is 'nominal':
    IG = Hy - sum(Hyx)
    use_threshold = None
  elif x_type is 'numeric':
    IG = max(split_info)
    use_threshold = splits[np.argmax( split_info )]

  return IG, use_threshold

"""
Function to Split Data Set on Best Feature
inputs:
  1) data (dict) to be split
  2) metadata
  3) best_feature (string) what feature to split on
  4) numeric_threshold (number/None) for split if feature is numeric
output: (dict), each element is branch name & associated data; ordered by program specifications
"""
def split_data(data, metadata, best_feature, numeric_threshold = None):

  # values of our best feature
  x = data[best_feature]
  m = metadata[best_feature]

  # dictionary for our two data splits
  split = {}

  # split data into groups for nominal data
  if m[0] is 'nominal':

    # for each value of feature, filter out data
    unique_x = m[1]
    for f in unique_x:
      sub_data = {}
      for k in data.keys():
        sub_data[k] = data[k][np.where(x == f)]
      split['%s = %s' % (best_feature, f)] = sub_data

  else:

    # filter out data less than threshold
    sub_data = {}
    for k in data.keys():
      sub_data[k] = data[k][np.where(x <= numeric_threshold)]
    split['%s <= %.6f' % (best_feature, numeric_threshold)] = sub_data

    # filter out data greater than threshold
    sub_data = {}
    for k in data.keys():
      sub_data[k] = data[k][np.where(x > numeric_threshold)]
    split['%s > %.6f' % (best_feature, numeric_threshold)] = sub_data

  # return split data as a dict
  return split

"""
Wrapper function to run splitting of data based on info gain
inputs:
  1) data (dict)
  2) metadata
  3) response_name: name of the response variable
output:
  1) info_values, information gains
  2) best feature, name of best feature
  3) splits, dict of split data
"""
def split_wrapper(data, metadata, response_name):

  # extract response values
  response_var = data[ response_name ]

  # compute info gain for each feature in data but not the response
  feature_names = []
  info_gain = []
  for feature in metadata.names():
    if feature is not response_name and feature in data.keys():
      feature_names.append(feature)
      info_gain.append(calc_info_gain(y = response_var, x = data[ feature ], x_type = metadata[ feature ][0]))

  # extract only the information gain
  info_values = np.array([x[0] for x in info_gain])

  # save the best feature (argmax takes first instance of max)
  best_feature_index = np.argmax(info_values)
  best_feature = feature_names[best_feature_index]
  numeric_threshold = info_gain[best_feature_index][1]

  # splits data into 2 or more splits and saves output into a thing
  splits = split_data(data, metadata, best_feature, numeric_threshold)

  # return results
  return info_values, best_feature, splits


"""
Function to build tree
inputs:
  1) data: (dict) keys are attributes and values are list of attribute values
  2) metadata: input data metadata
  3) response_name: (str) name of the response variable
  4) min_instances: (int) user defined m
  5) split_name: name of the preceding split
stopping criteria
  i) all training belongs to same class
  ii) fewer than m training instances
  iii) no feature has positive info gain
  iv) no more remaining candidate splits at node
"""
def build_tree(data, metadata, response_name, min_instances, split_name):

  # extract response values
  response_var = data[ response_name ]

  # stopping criteria:
    # (i) homogeneous responses
    # (ii) fewer than given m instances
    # (iii) no positive info gain
    # (iv) no more candidates to split on
  if len(set(response_var)) == 1 or len(response_var) < min_instances or len(data.keys()) == 1:
    l = node(node_type = 'leaf', node_name = split_name, data = data, metadata = metadata, responses = response_var)
    return l

  # run info gain calculations & split data
  info_values, best_feature, splits = split_wrapper(data, metadata, response_name)

  if np.all(info_values <= 0):
    l = node(node_type = 'leaf', node_name = split_name, data = data, metadata = metadata, responses = response_var)
    return l

  # generate a node to represent the split
  node_split = node(node_type = 'internal', node_name = split_name, data = data, metadata = metadata, feature_to_split = best_feature, splits = splits.keys())

  # for each split/leaf of current node, build additional leaves if applicable
  for k in splits.keys():

    # create a subnode to add to tree
    sub_node = build_tree(splits[k], metadata, response_name, min_instances, k)

    # if subnode is leaf and does not have a prediction then assign it
    if sub_node.get_type() is 'leaf' and sub_node.get_predictions() is None:
      sub_node.assign_predictions(response_var)
      # if still cannot predict, save to pass up
      if sub_node.get_predictions() is None:
        node_split.add_fix(sub_node)

    # # assign predictions to grandchildren
    grandchildren = sub_node.get_fix()
    if len(grandchildren) > 0:
      for v in grandchildren:
        v.assign_predictions(response_var)
        # if still cannot predict, save to pass up
        if v.get_predictions() is None:
          node_split.add_fix(v)

    # add the node to the list
    node_split.add_children(k, sub_node)

  # return node
  return node_split

"""
Function to build tree from root
"""
def build_tree_wrapper(data, metadata, min_instances):

  # run first iteration of finding best feature
  response_name = metadata.names()[-1]
  info_values, best_feature, splits = split_wrapper(data, metadata, response_name)
  root = node(node_type = 'internal', node_name = 'root', data = data, metadata = metadata, feature_to_split = best_feature, splits = splits)

  # populate tree
  for k in splits.keys():

    sub_node = build_tree(splits[k], metadata, response_name, min_instances, k)
    root.add_children(k, sub_node)

  # return root
  return root

"""
Function to predict given observations
"""
def predict(mod, data, nrow):

  predictions = []

  for i in range(nrow):

    n = mod
    while n.get_type() is 'internal':
      feature = n.get_feature()
      obs = data[feature][i]
      n = n.predict(obs)
    predictions.append( n.get_predictions() )

  return predictions

"""
Function to process training data
"""
def process_train(name, min_instances):

  d, m = parse_arff(name)
  root = build_tree_wrapper(d, m, min_instances)
  root.print_all(-1)

  return root

"""
Function to evaluate test data
"""
def process_test(name, mod):

  # open test data
  d, m = parse_arff(name)

  # extract actual responses
  actual = d[ m.names()[-1] ]
  nrow = len(actual)

  # predict responses with our tree
  predicted = predict(mod, d, nrow)

  # print output
  print "<Predictions for the Test Set Instances>"
  for i in range(nrow):
    print '%s: Actual: %s Predicted: %s' % (i+1, actual[i], predicted[i])
  print "Number of correctly classified: %s Total number of test instances: %s" % (sum(actual == predicted), nrow)


"""
Main function
"""
def make_tree(train, test, m):

  mod = process_train(train, m)
  process_test(test, mod)



if __name__ == '__main__':

  args = sys.argv
  make_tree(args[1], args[2], int(args[3]))

