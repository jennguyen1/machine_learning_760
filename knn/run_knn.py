
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
Function to find distance between two points
"""
def find_distance(a,b):
  return np.sqrt( sum((np.array(a) - np.array(b))**2) )

"""
Function to predict test from training
inputs:
  1) test: testing data
  2) train: training data
  3) k: k nearest neighbors
  4) y_type: response or class
  5) response_classses: if y_type is class, then the names of classes from arff file
"""
def predict(test, train, k, y_type, response_classes = None):

  # compute distances
  distances = np.array([])
  y = np.array([])
  for d in train:
    dist = find_distance(test, d[:(len(d)-1)])
    distances = np.append(distances, dist)
    y = np.append(y, d[(len(d)-1)])

  # order the list by distance
  order = np.argsort(distances)
  new_distances = distances[order]
  new_y = y[order]

  # top k responses
  top_distances = new_distances[:k]
  top_y = new_y[:k]

  # tie breaker for picking top y
  last_top_dist = top_distances[-1]

  # how many of the top distances to be the last top distance (ie how many to keep)
  how_many_to_keep = sum(top_distances == last_top_dist)

  # extract y corresponding to last top distance from k
  response_tie_breaker = y[distances == last_top_dist]

  # reformat responses if there are ties
  if len(Counter(response_tie_breaker)) > 1:
    top_y = np.append(top_y[:-how_many_to_keep], response_tie_breaker[:how_many_to_keep])

  # find prediction for knn
  if y_type == 'response':
    # compute mean of responses
    prediction = np.mean(top_y)

  else:
    # count and find the category that shows up most
    c = Counter(top_y)
    k = np.array(c.keys())
    v = np.array(c.values())
    i = np.where(v == max(v))[0]
    prediction = np.array(k[i])

    # tie-breaker
    if len(prediction) > 1:
      index = [response_classes.index(i) for i in prediction]
      prediction = prediction[np.argmin(index)]

    else:
      prediction = prediction[0]

  # return prediction
  return prediction

"""
Run KNN (p1)
"""
def knn(train, test, k, response_type, response_classes = None, print_output = False):

  output = ''

  # initialize pred/actual
  MAE = []
  n_correct = 0

  # predict
  for d in test:
    p = predict(d[:(len(d)-1)], train, k, response_type, response_classes)
    a = d[(len(d)-1)]

    if response_type == "response":
      MAE.append( np.absolute(p - d[(len(d)-1)]) )
      output += 'Predicted value : %.6f\tActual value : %.6f\n' % (p, a)
    else:
      n_correct += (p == a)
      output += 'Predicted class : %s\tActual class : %s\n' % (p, a)

  # print summary of algorithm
  if response_type == 'response':
    error_measure = MAE
    output += 'Mean absolute error : %.12f\n' % ( np.sum(MAE) / float(len(test)) )
  else:
    output += 'Number of correctly classified instances : %.0f\n' % n_correct

  # print total rows
  output += 'Total number of instances : %s\n' % str(len(test))

  # print accuracy for classification
  if response_type != 'response':
    error_measure = n_correct
    output += 'Accuracy : %.12f\n' % ( n_correct / float(len(test)) )

  # prints output if needed
  if print_output:
    print output

  # return results
  return error_measure

"""
Run KNN with selection (p2)
"""
def knn_select(train, test, ks, response_type, response_classes = None):

  k_error = []
  for k in ks:

    error = []

    # LOOCV iterations
    for i in range(len(train)):
      new_train = [train[l] for l in range(len(train)) if l != i]
      new_test = [train[i]]

      # run knn depending on whether have classes or not
      e = knn(new_train, new_test, k, response_type, response_classes, False)

      # add error for LOOCV iteration
      error.append(e)

    # compute average error across iterations
    if response_type == 'response':
      mean_err = np.mean(error)
    else:
      mean_err = len(error) - np.sum(error)
    k_error.append(mean_err)

    # print out terms
    if response_type == 'response':
      prefix = 'Mean absolute error for k = %.0f : %.12f'
    else:
      prefix = 'Number of incorrectly classified instances for k = %.0f : %.0f'
    best_k = ks[np.argmin(k_error)]
    print prefix % (k, mean_err)

  # use best k
  print 'Best k value : %.0f' % best_k

  # run knn with best k
  knn(train, test, best_k, response_type, response_classes, True)

"""
Run all
"""
def run_all(train_file, test_file, k, kind):

  # open train & test data
  train_data, train_meta = parse_arff(train_file)
  test_data, test_meta = parse_arff(test_file)

  # extract response type
  response_type = train_meta.names()[-1]

  # run knn depending on response type & selection or not
  if response_type == 'response':
    if kind == 'knn-select':
      knn_select(train_data, test_data, k, response_type, None)
    else:
      print 'k value : %d' % k
      knn(train_data, test_data, k, response_type, None, True)

  else:
    if kind == 'knn-select':
      knn_select(train_data, test_data, k, response_type, train_meta['class'][1])
    else:
      print 'k value : %d' % k
      knn(train_data, test_data, k, response_type, train_meta['class'][1], True)


if __name__ == '__main__':

  args = sys.argv

  # name of function
  kind = args[1]

  # train/test files
  train_file = args[2]
  test_file = args[3]

  # assign k based on name of function
  if kind == 'knn-select':
    k = np.array([int(args[4]), int(args[5]), int(args[6])])
  else:
    k = int(args[4])

  # run function
  run_all(train_file, test_file, k, kind)

