
import scipy.io.arff as arff
import numpy as np
import pandas as pd
from collections import OrderedDict
from collections import Counter
from itertools import product
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

  # save data into row by row format
  row_data = [[v if type(v) is np.string_ else round(v, 14) for v in l] for l in raw_data]

  # save data into dictionary
  feature_data = {}
  for i in range(len(metadata.names())):
    feature = [l[i] for l in raw_data]
    feature_data[metadata.names()[i]] = np.array(feature)

  return feature_data, row_data, metadata

"""
Drop all duplicates
"""
def drop_all(df2, df1):
  return df2[~df2.isin(df1).all(1)]

"""
Generates grid from dictionary
"""
def expand_grid(dictionary):
   return pd.DataFrame([row for row in product(*dictionary.values())], columns=dictionary.keys())

"""
Compute conditional probabilities
"""
def cond_probs(df, all_combos, group_vars, condition_vars, target_var, debug = False):

  # counts of x's
  counts = df.groupby(group_vars).size().reset_index()
  counts.columns = sum( [list(counts.columns)[:-1], ['count']], [] )

  # fix 0 counts
  extra = drop_all(all_combos, counts[group_vars])
  if extra.shape[0] > 0:
    extra['count'] = 0
    counts = pd.concat([counts, extra])

  # total y
  totals = df.groupby(condition_vars).size().reset_index()

  # combine and calculate MAP conditional probabilities
  m = counts.merge(totals, left_on = condition_vars, right_on = condition_vars, how = 'outer')
  m.fillna(0, inplace=True)
  m.columns = sum( [list(m.columns)[:-1], ['total']], [] )
  m['p'] = (m['count'] + 1) / (m['total'] + all_combos[target_var].drop_duplicates().shape[0] )
  out_df = m[sum( [group_vars, ['p']], [] )]

  # return results
  return out_df


"""
Naive Bayes
"""
class nb:

  """
  Compute probabilities with MLE and laplace estimates
  """
  def __init__(self, d, m):

    # feature names
    self.features = m.names()
    self.features.pop()

    # put data into data frame
    df = pd.DataFrame(d)
    self.probabilities = {}

    # loop through columns
    for k in df.columns:

      # P(Y =y)
      if k == 'class':
        out_df = df.groupby('class').apply(lambda x: (len(x) + 1) / float(df.shape[0] + 2)).reset_index()
        out_df.columns = ['class', 'p']

      # P(X = x | Y = y)
      else:
        all_combos = expand_grid({'class': list(m['class'][1]), k: list(m[k][1])})
        out_df = cond_probs(df = df, all_combos = all_combos, group_vars = ['class', k], condition_vars = ['class'], target_var = [k])

      # save output table
      self.probabilities[k] = out_df

  """
  Get desired probabilties
  """
  def get_cond_probs(self, y, x, x_value):
    df = self.probabilities[x]
    return df[(df['class'] == y) & (df[x] == x_value)].reset_index(drop = True).loc[0, 'p']

  """
  Predict with probability
  """
  def predict(self, instance):

    compute = {}

    # P(Y = y | x)
    for y in list(self.probabilities['class']['class']):
      df = self.probabilities['class']
      compute[y] = df[df['class'] == y].reset_index(drop = True).loc[0, 'p']
      for i, k in enumerate(self.features):
        compute[y] *= self.get_cond_probs(y, k, instance[i])

    k, v =  compute.items()[0]
    p_y_x = v / float(sum(compute.values()))

    # assign prediction
    if p_y_x > 0.5:
      prediction = k
      prob = p_y_x
    else:
      prediction = compute.keys()[1]
      prob = 1 - p_y_x

    return prediction, prob

  """
  Predict for all of test set
  """
  def predict_all(self, test):

    for k in self.features:
      print '%s %s' % (k, 'class')
    print ''
    accurate = 0

    for r in test:
      actual = r[len(r)-1]
      predict, prob =  self.predict(r)
      if actual.startswith("'") and actual.endswith("'"):
        actual = actual[1:-1]
        predict = predict[1:-1]
      print '%s %s %.12f' % (predict, actual, prob)
      accurate += (actual == predict)

    print ''
    print accurate

"""
TAN
"""
class tan:

  """
  Initiate class with edge values
  """
  def __init__(self, d, m):

    # feature names
    self.features = m.names()
    self.features.pop()

    # initiate data
    df = pd.DataFrame(d)

    # initialize edges
    self.edges = {}
    for f in self.features:
      self.edges[f] = {}

    # loop through pairwise of features
    for a in range(len(self.features)):
      for b in range(a+1, len(self.features)):

        i = self.features[a]
        j = self.features[b]

        # make a data frame of values
        i_df = df[[i, j, 'class']]
        all_combos = expand_grid({'class': list(m['class'][1]), i: list(m[i][1]), j: list(m[j][1])})

        # get conditional probabilities
        tab_cond_x1 = cond_probs(df = i_df, all_combos = all_combos[['class', i]].drop_duplicates(), group_vars = ['class', i], condition_vars = ['class'], target_var = [i])
        tab_cond_x2 = cond_probs(df = i_df, all_combos = all_combos[['class', j]].drop_duplicates(), group_vars = ['class', j], condition_vars = ['class'], target_var = [j])
        tab_cond_x1x2 = cond_probs(df = i_df, all_combos = all_combos, group_vars = ['class', i, j], condition_vars = ['class'], target_var = [i, j])

        # get intersecting probabilities
        # counts of x's
        counts = i_df.groupby(['class', i, j]).size().reset_index()
        counts.columns = ['class', i, j, 'count']

        # fix 0 counts
        extra = drop_all(all_combos, counts[['class', i, j]])
        if extra.shape[0] > 0:
          extra['count'] = 0
          counts = pd.concat([counts, extra])

        # total for denominator
        counts['total'] = i_df.shape[0]

        # combine and calculate MAP conditional probabilities
        counts['p'] = (counts['count'] + 1) / (counts['total'] + all_combos.shape[0] )
        tab_all = counts[ ['class', i, j, 'p'] ]

        # loop through individual values
        I = 0
        for y in list(m['class'][1]):
          for x1_val in list(m[i][1]):
            for x2_val in list(m[j][1]):

              # individual probabilities
              p_x1x2y = tab_all[ (tab_all[i] == x1_val) & (tab_all[j] == x2_val) & (tab_all['class'] == y) ].reset_index(drop = True).loc[0, 'p']
              p_x1_y = tab_cond_x1[ (tab_cond_x1[i] == x1_val) & (tab_cond_x1['class'] == y) ].reset_index(drop = True).loc[0, 'p']
              p_x2_y = tab_cond_x2[ (tab_cond_x2[j] == x2_val) & (tab_cond_x2['class'] == y) ].reset_index(drop = True).loc[0, 'p']
              p_x1x2_y = tab_cond_x1x2[ (tab_cond_x1x2[i] == x1_val) & (tab_cond_x1x2[j] == x2_val) & (tab_cond_x1x2['class'] == y) ].reset_index(drop = True).loc[0, 'p']

              # add values to overall mutual information
              I += p_x1x2y * np.log2(p_x1x2_y/(p_x1_y*p_x2_y))

        # assign multual information edges
        self.edges[i][j] = I
        self.edges[j][i] = I

    # run prim
    self.prim()

    # calculate probabilities
    self.bayes_net(df, m)

  """
  Prim's algorithm
  """
  def prim(self):

    v_new = [self.features[0]]
    e_new = []

    while True:
      choose_next = pd.DataFrame({'u': ['u'], 'v': ['v'], 'weight': [0]})

      # collect the weights
      for u in v_new:
        v = self.edges[u].keys()
        next_weight = self.edges[u].values()
        choose_next = pd.concat([choose_next, pd.DataFrame({'u': u, 'v': v, 'weight': next_weight})])
        choose_next = choose_next[~choose_next['v'].isin(v_new)]

      # break out
      if choose_next.shape[0] == 1:
        break

      # tie breaker
      def tie_breaker(df):
        for f in self.features:
          if f in list(df['u']):
            for k in self.features:
              if k in list(df['v']):
                return df[ (df['u'] == f) & (df['v'] == k) ].reset_index()

      # extract the top edge
      top_edges = tie_breaker(choose_next[choose_next['weight'] == max(choose_next['weight'])].reset_index())

      # add top edge to the list
      v_new.append(top_edges.loc[0, 'v'])
      e_new.append(top_edges)

    # output final results
    self.edge_directions = pd.concat(e_new).sort(['u'])[['u', 'v', 'weight']]

  """
  Find conditional probabilities
  """
  def bayes_net(self, df, m):

    self.probabilities = {}

    # P(Y = y)
    self.probabilities['class'] = df.groupby('class').apply(lambda x: (len(x) + 1) / float(df.shape[0] + 2)).reset_index()
    self.probabilities['class'].columns = ['class', 'p']

    # P(X = x | Y = y) root
    root = self.features[0]
    all_combos = expand_grid({'class': list(m['class'][1]), root: list(m[root][1])})
    self.probabilities[root] = cond_probs(df = df, all_combos = all_combos, group_vars = ['class', root], condition_vars = ['class'], target_var = [root])

    # P(X = x | Y = y, etc)
    for i, r in self.edge_directions.iterrows():

      # find directions
      conditioning_var = r[0]
      target_var = r[1]

      # calculate var and add to structure
      all_combos = expand_grid({'class': list(m['class'][1]), conditioning_var: list(m[conditioning_var][1]), target_var: list(m[target_var][1])})
      self.probabilities[target_var] = cond_probs(df = df, all_combos = all_combos, group_vars = ['class', conditioning_var, target_var], condition_vars = ['class', conditioning_var], target_var = [target_var])

  """
  Get desired probabilties
  """
  def get_cond_probs(self, y, x, x_value, instance):

    # set up original data
    other_data = pd.DataFrame({'var': self.features, 'value': instance})

    # find probability
    df = self.probabilities[x]
    cond_var = list(set(df.columns[:3]) - set(['class', x, 'p']))
    if len(cond_var) > 0:
      cond_var_value = other_data[other_data['var'] == cond_var[0]].reset_index(drop = True).loc[0,'value']

    # find probability
    if x == self.features[0]:
      r = df[(df['class'] == y) & (df[x] == x_value)].reset_index(drop = True).loc[0, 'p']
    else:
      r = df[(df['class'] == y) & (df[x] == x_value) & (df[cond_var[0]] == cond_var_value)].reset_index(drop = True).loc[0, 'p']

    return r

  """
  Predict with probability
  """
  def predict(self, instance):

    instance.pop()
    compute = {}

    # P(Y = y | x)
    for y in list(self.probabilities['class']['class']):
      df = self.probabilities['class']
      compute[y] = df[df['class'] == y].reset_index(drop = True).loc[0, 'p']
      for i, k in enumerate(self.features):
        compute[y] *= self.get_cond_probs(y, k, instance[i], instance)

    k, v =  compute.items()[0]
    p_y_x = v / float(sum(compute.values()))

    # assign prediction
    if p_y_x > 0.5:
      prediction = k
      prob = p_y_x
    else:
      prediction = compute.keys()[1]
      prob = 1 - p_y_x

    return prediction, prob

  """
  Predict for all of test set
  """
  def predict_all(self, test):

    for k in self.features:
      if k == self.features[0]:
        print '%s %s' % (k, 'class')
      else:
        print '%s %s %s' % (k, self.edge_directions[self.edge_directions['v'] == k].reset_index(drop = True).loc[0, 'u'], 'class')
    print ''
    accurate = 0

    for r in test:
      actual = r[len(r)-1]
      predict, prob =  self.predict(r)
      if actual.startswith("'") and actual.endswith("'"):
        actual = actual[1:-1]
        predict = predict[1:-1]
      print '%s %s %.12f' % (predict, actual, prob)
      accurate += (actual == predict)

    print ''
    print accurate


if __name__ == '__main__':

  args = sys.argv

  # train/test files
  train_file = args[1]
  test_file = args[2]
  train_d, trash, train_m = parse_arff(train_file)
  trash, test_d, test_m = parse_arff(test_file)

  # run functions
  kind = args[3]
  if kind == 'n':
    nbayes = nb(train_d, train_m)
    nbayes.predict_all(test_d)
  else:
    t = tan(train_d, train_m)
    t.predict_all(test_d)

