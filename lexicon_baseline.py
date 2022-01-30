import pandas as pd
from nltk.corpus import wordnet as wn
import csv
from sklearn.metrics import classification_report

# # # code snippet taken from https://github.com/cltl/lexical-negation-dictionary

# extract all direct antonym word pairs from wordnet
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

def create_negation_lexicon(inputfile_antonyms):
    '''
    This function extracts all affixal negation cues from a file containing antonym word pairs, and creates a list of affixal negations
    It then combines the affixal negations with a list of common negation expressions to form an exhaustive lexicon of 1761 affixal and lexical negations
    
    :param inputfile: .csv file of antonym word pairs extracted from wordnet by van Son et al. (2016)
    :returns: a list 
    '''
    # open created .csv file with all antonym pairs in pandas df
    wordnet_df = pd.read_csv(inputfile_antonyms, sep="\t", header=0)

    # define needed lists to iterate over
    pos_antonyms = wordnet_df['pos_element'].tolist()
    neg_antonyms = wordnet_df['neg_element'].tolist()
    prefixes = ['un', 'dis', 'im', 'in', 'non', 'ir']
    suffixes = ['less', 'lessly', 'lessness']
    neg_exp = ['nor', 'neither', 'without', 'nobody', 'none', 'nothing', 'never', 'not', 'no', 'nowhere', 'non']

    # iterate over all antonyms and extract affixal negations
    # if a word in the wordpair starts with a negational prefix AND the word without the affix matches its antonym pair: append word to affixalnegs list
    affixalnegs = [] # define affixal negations list
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
        # for suffixes: if word ends with 'less', 'lessly' or 'lessness' it can be assumed to be an affixal negation and is appended to affixalnegs list
        # exception: the word 'bless' is not a negation and is excluded in hardcode
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

    # add common lexical negations to affixal negations to create full negation lexicon
    neg_lexicon = affixalnegs + neg_exp
    
    return neg_lexicon

def rule_based_negation(inputfile, neg_lexicon):
    '''
    This function contains a rule-based negation system. It's main working mechanism is a lexicon look-up.
    '''
    df = pd.read_csv(inputfile, sep="\t", header=None, names=["Book", "Sent nr", "Token nr", "Token", "Label"])

    # put all tokens into list:
    tokenlist = df['Token'].tolist()

    # extract gold list for evaluation
    y_true = df['Label'].tolist()

    # rule-based system
    y_pred = []

    i = 0
    for token in tokenlist:
        token = token.lower()
        if len(y_pred) == i:  
            if token == 'by' and tokenlist[i+1] == 'no' and tokenlist[i+2] == 'means':
                y_pred.append('B-NEG')
                y_pred.append('I-NEG')
                y_pred.append('I-NEG')
            elif token == 'on' and tokenlist[i+1] == 'the' and tokenlist[i+2] == 'contrary':
                y_pred.append('B-NEG')
                y_pred.append('I-NEG')
                y_pred.append('I-NEG')
            elif token == 'not' and tokenlist[i+1] == 'for' and tokenlist[i+2] == 'the' and tokenlist[i+3]== 'world':
                y_pred.append('B-NEG')
                y_pred.append('I-NEG')
                y_pred.append('I-NEG')
                y_pred.append('I-NEG')
            elif token == 'rather' and tokenlist[i+1] == 'than':
                y_pred.append('B-NEG')
                y_pred.append('I-NEG')
            elif token == 'nothing' and tokenlist[i+1] == 'at' and tokenlist[i+2] == 'all':
                y_pred.append('B-NEG')
                y_pred.append('I-NEG')
                y_pred.append('I-NEG')
            elif token in neg_lexicon:
                y_pred.append('B-NEG')
            else:
                y_pred.append('O')
            i += 1
        else:
            i += 1
            continue
            
    return y_pred, y_true


# function call to create negation lexicon
inputfile_antonyms = 'antonyms.csv'
neg_lexicon = create_negation_lexicon(inputfile_antonyms)

# check negation lexicon length
print('length of negation cue lexicon:')
print(len(neg_lexicon))

# open training and dev data
inputfile_train = 'data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt'
inputfile_dev = 'data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt'

# function calls for rule-based negation classifying system
train_y_pred, train_y_true = rule_based_negation(inputfile_train, neg_lexicon)
dev_y_pred, dev_y_true = rule_based_negation(inputfile_dev, neg_lexicon)

# compare rule-based predictions with annotated labels from data
train_report = classification_report(train_y_true, train_y_pred, digits = 7, zero_division=0)
dev_report = classification_report(dev_y_true, dev_y_pred, digits = 7, zero_division=0)

#print classification reports
print()
print('training data rule-based classification report')
print(train_report)
print()
print('dev data rule-based classification report')
print(dev_report)
