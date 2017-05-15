import numpy as np
import sklearn as sk
import argparse
import conllu
import json

from sklearn.externals import joblib
from collections import defaultdict
from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegressionCV
#from sklearn.svm import SVC
from scipy.sparse import csr_matrix, vstack



# read the labels
def read_labels(infile):
  with open(infile, encoding='utf-8') as f:
    labels = json.load(f)

  print("Loaded labels from {file}".format(file=infile))
  return labels



def to_one_hot(index, length):
  "generate a one hot vector"
  v = np.zeros(length, dtype='uint8')
  try:
    v[index] = 1
  except IndexError:
    pass
  return v



def featurize_morph(morph, morph_all):
  vec = []
  for feature, poss_values in sorted(morph_all.items()):
    small_vec = [0] * len(poss_values)

    if feature in morph:
      for index, value in enumerate(poss_values):
        if value in morph[feature]:
          small_vec[index] = 1

    vec += small_vec

  #return lil_matrix(vec, (1, len(vec)), dtype='uint8')
  return np.array(vec, dtype='uint8')



# write the labels
def write_labels(outfile, labels):
  with open(outfile, 'w') as f:
    #print(labels)
    #json.dump(labels, f, ensure_ascii=False)
    json.dump(labels, f, encoding='utf-8', ensure_ascii=False)

  print("Saved labels as {file}".format(file=outfile))



# train with liblinear
def train(instances, labels):
  clf = LinearSVC()
  clf.fit(instances, labels)
  return clf



# serialize the model
def dump_model(clf, outfile):
  joblib.dump(clf, outfile)
  print("Saved model  : {file}".format(file=outfile))



# deserialize the model
def load_model(infile):
  clf = joblib.load(infile)
  print("Loaded model : {file}".format(file=infile))
  return clf



# predict with liblinear
def predict(clf, instances):
  return clf.predict(instances)



# write labeled conllu file
def write_conllu(data, predictions, outfile):
  "write CoNLLU file one line at a time"
  i = 0
  sentences = []
  for sent in data.gen_sentences(include_multiwords=True):
    sent_out = []
    for token in sent:
      if type(token.ID) is int:
        sent_out.append(token._replace(DEPREL=predictions[i]))
        i += 1
      else:
        sent_out.append(token)

    sentences.append(sent_out)

  conllu.Dataset(outfile).write_sentences(sentences)
  print("Wrote output : {file}".format(file=outfile))



def build_dicts(data):
  "build the dictionaries for the features"
  form_counts = defaultdict(lambda: 0)
  lemma_counts = defaultdict(lambda: 0)

  morph = defaultdict(set)

  uposes = set()
  deprels = set()

  forms = defaultdict(lambda: len(forms))
  lemmas = defaultdict(lambda: len(lemmas))
  forms['_']
  lemmas['_']
  for sent in data.gen_sentences():
    for token in sent:
      form_counts[token.FORM] += 1
      lemma_counts[token.LEMMA] += 1
      uposes.add(token.UPOSTAG)
      deprels.add(token.DEPREL)
      for key, values in token.FEATS.items():
        for value in values:
          morph[key].add(value)

  for form, count in form_counts.items():
    if count > 1:
      forms[form]

  for lemma, count in lemma_counts.items():
    if count > 1:
      lemmas[lemma]

  morph = { key: tuple(sorted(value)) for key, value in morph.items() }

  uposes = { upos : index + 1 for index, upos in enumerate(uposes) }
  uposes['_'] = 0
  deprels = { deprel : index for index, deprel in enumerate(deprels) }

  return { 'forms' : dict(forms), 'lemmas' : dict(lemmas), 'morph' : morph, 'uposes' : uposes, 'deprels' : deprels }



def featurize(data, labels, include_targets=False):
  "extract the features from the data"
  forms = labels['forms']
  lemmas = labels['lemmas']
  uposes = labels['uposes']
  deprels = labels['deprels']
  morph = labels['morph']

  X = []
  y = []
  for sent in data.gen_sentences():
    for token in sent:
      if type(token.HEAD) is int and token.HEAD != 0:
        head = sent[token.HEAD - 1]
      else:
        head = conllu.Word._make([0,'_','_','_',None,{},None,None,None,None])
      
      if type(token.ID) is int and token.ID > 0:
        prev = sent[token.ID - 1]
      else:
        prev = conllu.Word._make([0,'_','_','_',None,{},None,None,None,None])
      
      if type(token.ID) is int and token.ID < len(sent) - 1:
        next = sent[token.ID + 1]
      else:
        next = conllu.Word._make([0,'_','_','_',None,{},None,None,None,None])

      v = np.concatenate([
      #v = vstack([
        # token id
        #np.array([token.ID], dtype='uint8'),
        # token upos
        to_one_hot(uposes[token.UPOSTAG] if token.UPOSTAG in uposes else 0, len(uposes)),
        # token form
        to_one_hot(forms[token.FORM] if token.FORM in forms else 0, len(forms)),
        # token form
        to_one_hot(lemmas[token.LEMMA] if token.LEMMA in lemmas else 0, len(lemmas)),
        # token morph
        featurize_morph(token.FEATS, morph),
        # token context
        to_one_hot(uposes[prev.UPOSTAG] if prev.UPOSTAG in uposes else 0, len(uposes)),
        to_one_hot(uposes[next.UPOSTAG] if next.UPOSTAG in uposes else 0, len(uposes)),
        # head - token distance
        #np.array([token.HEAD - token.ID if token.HEAD != 0 else 0], dtype='uint8'),

        # head id
        #np.array([head.ID], dtype='uint8'),
        # head upos
        to_one_hot(uposes[head.UPOSTAG] if head.UPOSTAG in uposes else 0, len(uposes)),
        # head form
        to_one_hot(forms[head.FORM] if head.FORM in forms else 0, len(forms)),
        # head lemma
        to_one_hot(lemmas[head.LEMMA] if head.LEMMA in lemmas else 0, len(lemmas)),
        # head morph
        featurize_morph(head.FEATS, morph)
      ])

      #v /= np.linalg.norm(v)
      X.append(csr_matrix(v, dtype='uint8'))
      if include_targets:
        y.append(deprels[token.DEPREL])

  X = vstack(X)

  if include_targets:
    return X, y
  else:
    return X



# main method
def main():

  parser = argparse.ArgumentParser(
    description="Train and predict CoNNLU dependency relations.")

  parser.add_argument("-i", "--input", required=True)
  parser.add_argument("-o", "--output")
  parser.add_argument("-m", "--model", required=True)
  parser.add_argument("-l", "--labels", required=True)
  parser.add_argument("-t", "--train", action='store_true')

  args = parser.parse_args()

  print()
  print("Input  : {file}".format(file=args.input))
  print("Output : {file}".format(file=args.output))
  print("Model  : {file}".format(file=args.model))
  print("Labels : {file}".format(file=args.labels))
  print()

  # read the unlabeled CoNLLU input file
  conllu_data = conllu.Dataset(args.input)

  print("Read input   : {file}".format(file=args.input))

  if args.train:
    labels = build_dicts(conllu_data)
    write_labels(args.labels, labels)
  else: # predict
    labels = read_labels(args.labels)

  # create the instance data
  if args.train:
    X, y = featurize(conllu_data, labels, include_targets=True)
  else:
    X = featurize(conllu_data, labels)

  print("Featurized the data.")

  if args.train:
    clf = train(X, y)
    dump_model(clf, args.model)
  else: # predict
    clf = load_model(args.model)
    rev_d = { value : key for key, value in labels['deprels'].items() }
    predictions = [ rev_d[x] for x in predict(clf, X) ]
    write_conllu(conllu_data, predictions, args.output)

  print("--- Done ---")



if __name__ == '__main__':
  main()

