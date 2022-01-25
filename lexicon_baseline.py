import pandas as pd
from nltk.corpus import wordnet as wn
import csv
from sklearn.metrics import classification_report

# # # code snippet taken from https://github.com/cltl/lexical-negation-dictionary

antonyms = {}
for synset in wn.all_synsets():
    # if synset.pos() in ['s', 'a']: # If synset is adj or satelite-adj.
    for lemma in synset.lemmas():
        if lemma.antonyms():
            for antonym in lemma.antonyms():
                pair = (lemma.name(), antonym.name())
                reversed_pair = (antonym.name(), lemma.name())
                if not reversed_pair in antonyms:
                    antonyms[pair] = [lemma.name(), antonym.name(), synset.pos(), lemma.key(), antonym.key(),
                                      synset.definition(), antonym.synset().definition()]

# write antonym pairs to .csv
outfilename = "antonyms.csv"
with open(outfilename, "w") as outfile:
    csvreader = csv.writer(outfile, delimiter="\t")
    header = ["pos_element", "neg_element", "POS", "pos_key", "neg_key", "pos_definition", "neg_definition"]
    csvreader.writerow(header)
    for pair in antonyms:
        row = antonyms[pair]
        csvreader.writerow(row)
        
# # # end of code snippet 

# open created .csv file with all antonym pairs in pandas df
inputfile = 'antonyms.csv'
wordnet_df = pd.read_csv(inputfile, sep="\t", header=0)

# define needed lists to iterate over
pos_antonyms = wordnet_df['pos_element'].tolist()
neg_antonyms = wordnet_df['neg_element'].tolist()
prefixes = ['un', 'dis', 'im', 'in', 'non', 'ir']
suffixes = ['less', 'lessly', 'lessness']
neg_exp = ['nor', 'neither', 'without', 'nobody', 'none', 'nothing', 'never', 'not', 'no', 'nowhere', 'non']

# iterate over all antonyms and extract affixal negations
affixalnegs = [] # define negation cue lexicon list
i = 0
for antonym in neg_antonyms:
    pos_antonym = pos_antonyms[i]
    for prefix in prefixes:
        if antonym.startswith(prefix):
            len_prefix = len(prefix)
            split_antonym = antonym[len_prefix:] 
            if pos_antonym == split_antonym:
                if antonym not in affixalnegs:
                    affixalnegs.append(antonym)
        elif pos_antonym.startswith(prefix):
            split_antonym = pos_antonym[len_prefix:] 
            if antonym == split_antonym:
                if pos_antonym not in affixalnegs:
                    affixalnegs.append(pos_antonym)
    for suffix in suffixes:
        if antonym.endswith(suffix):
            if antonym != 'bless':
                if antonym not in affixalnegs:
                    affixalnegs.append(antonym)
        elif pos_antonym.endswith(suffix):
            if pos_antonym != 'bless':
                if pos_antonym not in affixalnegs:
                    affixalnegs.append(pos_antonym)
        
    i += 1

# add lexical negations to affixal negations to create full negation lexicon
affixalnegs = affixalnegs + neg_exp

# check negation lexicon length
print('length of negation cue lexicon:')
print(len(affixalnegs))

# rule-based classifying system
# open training data in pandas df
inputfile_train = 'SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt'
df = pd.read_csv(inputfile_train, sep="\t", header=None, names=["Book", "Sent nr", "Token nr", "Token", "Label"])

# put all tokens and labels into a list:
tokenlist = df['Token'].tolist()
label_list = df['Label'].tolist() 

# if token occurs on negation cue lexicon, label token as 'B-NEG'
# otherwise, label token as 'O'
y_pred = []
for token in tokenlist:
    if token in affixalnegs:
        y_pred.append('B-NEG')
    else:
        y_pred.append('O')
      
# compare rule-based predictions with annotated labels from data
y_true = label_list
report = classification_report(y_true,y_pred,digits = 7, zero_division=0)

print('training data classification report')
print(report)
