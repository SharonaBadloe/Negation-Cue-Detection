# imports
import pandas as pd
import nltk
import spacy
import string
nlp = spacy.load("en_core_web_sm")

# filepaths 
inputfile_train = 'c:/Users/desir/Desktop/text_mining/applied TM/bioscope.papers.columns.txt'


def add_features(inputfile):
    '''
    Takes as input a .txt file consisting of a tokenized corpus. Converts the corpus into a pandas df for feature extracting.
    Extracts the following features from the data:
    Token i-1
    Token i+1
    Lemma i
    Lemma i-1
    Lemma i+1
    POS i
    POS i-1
    POS i+1
    POS-tag classification
    Punctuation
    MatchesNegExp
    hasNegAffix
    Negated event
    Negational affix
    :param inputfile: .txt file
    :returns: .conll file with added features
    '''
    # open inputfile in pandas dataframe
    # add column headers
    df = pd.read_csv(inputfile, sep="\\t", header=None, names=["doc id", "Sent id", "Token id", "Token", "Label", 'scope label'])
    
    df.fillna("")
    # put all our tokens into a list:
    tokenlist = df['Token'].tolist()

    # put all labels into a list
    label_list = df['Label'].tolist()

    # extract previous token and next token
    # assign token lists
    pretoken = []
    nexttoken = []

    # iterate over tokenlist and get the previous and next tokens by index
    i = 0
    for token in tokenlist:
        if i == 0:
            pretoken.append('<BOD>')
            nexttoken.append(tokenlist[i+1])
        elif i == len(tokenlist) - 1:
            pretoken.append(tokenlist[i-1])
            nexttoken.append('<EOD>')
        else:
            pretoken.append(tokenlist[i-1])
            nexttoken.append(tokenlist[i+1])
        i += 1

    # add token features to dataframe
    df['Pre token'] = pretoken
    df['Next token'] = nexttoken

    # extract lemma
    # iterate over tokens and extract lemma
    d = []
    for token in tokenlist:
        doc = nlp(token)
        d.append(doc[0].lemma_)
    df['Lemma'] = d     # add lemma to dataframe

    # extract previous and next lemma
    lemmalist = df['Lemma'].tolist()
    pre_lemma = []
    next_lemma = []

    # iterate over lemma list to get previous and next lemma by index
    i = 0
    for lemma in lemmalist:
        if i == 0:
            pre_lemma.append('<BOD>')
            next_lemma.append(lemmalist[i+1])
        elif i == len(lemmalist) - 1:
            pre_lemma.append(lemmalist[i-1])
            next_lemma.append('<EOD>')
        else:
            pre_lemma.append(lemmalist[i-1])
            next_lemma.append(lemmalist[i+1])
        i += 1

    # add previous and next lemma to dataframe
    df['Pre lemma'] = pre_lemma
    df['Next lemma'] = next_lemma

    # extract pos tags
    d = []
    for token, pos in nltk.pos_tag(tokenlist):
        d.append(pos)
    df['POS'] = d   # add postags to dataframe

    # extract previous and next postags
    # create two lists for collecting previous tag and next tag
    poslist = df['POS'].tolist()
    pre_pos = []
    next_pos = []

    # loop over tag list, extract previous and next tag
    i = 0
    for pos in poslist:
        if i == 0:
            pre_pos.append('<BOD>')
            next_pos.append(poslist[i+1])
        elif i == len(tokenlist) - 1:
            pre_pos.append(poslist[i-1])
            next_pos.append('<EOD>')
        else:
            pre_pos.append(poslist[i-1])
            next_pos.append(poslist[i+1])
        i += 1

    # add previous and next pos to dataframe
    df['Pre POS'] = pre_pos
    df['Next POS'] = next_pos

    # code for classifying POS-tags
    pos_cat_list = []
    for tag in poslist:
        if tag == "VBZ" or tag == "VBP" or tag == "VBZ" or tag == "VBN" or tag == "VBG" or tag == "VB" or tag == "VBD" or tag == "MD":
            pos_cat_list.append("VB")
        elif tag == "PRP$" or tag =="PRP" or tag =="POS" or tag =="WP":
            pos_cat_list.append("PRO")
        elif tag == "EX" or tag == "DT" or tag == "CD" or tag == "LS" or tag == "PDT" or tag == "WDT" or tag == "UH" or tag == "TO":
            pos_cat_list.append("OTH")
        elif tag == "RP" or tag == "RBS" or tag == "RBR" or tag == "CC" or tag == "RB" or tag == "WRB":
            pos_cat_list.append("ADVB")
        elif tag == "NNS" or tag == "NNP" or tag =="NN":
            pos_cat_list.append("NN")
        elif tag == "JJS" or tag == "JJR" or tag == "JJ" or tag == "IN":
            pos_cat_list.append("ADJ")
        #if anything else, not seen in trainingdata comes forward:
        else:
            pos_cat_list.append("OTH")
    df['POS_classified'] = pos_cat_list

    #code for extracting punctuation marks
    d = []
    print("processing words with spacy")
    for word in tokenlist:
        doc = nlp(word)
        punctuation_list = []
        for item in doc:
            if item.is_punct == True:
                punctuation_list.append("1")
            elif item.is_punct == False:
                punctuation_list.append("0")
            else:
                print("Error: check word processing")
                break
        d.append(punctuation_list)
    df['Punctuation_spacy'] = d

    #extracting punctuation marks without the use of spacy

    d = []
    punct_list = string.punctuation
    for token in df['Token']:
        if token in punct_list:
            d.append(1)
        else:
            d.append(0)
    df['Punctuation_python'] = d

    # extract MatchesNegExp feature
    # define NegExpList
    neg_exp = ['nor', 'neither', 'without', 'nobody', 'none', 'nothing', 'never', 'not', 'no', 'nowhere', 'non']

    # iterate over tokens and check if they match the NegExpList
    matchesnegexp = []
    for token in tokenlist:
        if token in neg_exp:
            matchesnegexp.append(1)
        else:
            matchesnegexp.append(0)
    df['MatchesNegExp'] = matchesnegexp  # add feature to dataframe

    # extract negational affix features
    # create list of positive vocabulary
    positives = []
    i = 0
    for item in label_list:
        if item == 'O':
            positives.append(tokenlist[i].lower())  # lowercase tokens for lookup
        i += 1

    # assign lists for each of the three features
    hasnegaffix = []
    negated_event = []
    neg_affix = []

    # iterate over each token
    for token in tokenlist:
        token = token.lower()       # lowercase tokens for lookup
        if token.startswith('un'):    # if token has negational affix:
            split_token = token[2:]   # split affix from token
            if token not in positives:   # if token is not a positive word:
                negated_event.append(split_token)    #  add negated event to list
                neg_affix.append('un')        # add affix to affix list
                hasnegaffix.append(1)         #  add 1 to binary affix list
            else:
                negated_event.append('-')   # if token is a positive word:
                neg_affix.append('-')       # add dashes and 0 to appropriate lists
                hasnegaffix.append(0)
        elif token.startswith('dis'):
            split_token = token[3:]
            if token not in positives:
                negated_event.append(split_token)
                neg_affix.append('dis')
                hasnegaffix.append(1)
            else:
                negated_event.append('-')
                neg_affix.append('-')
                hasnegaffix.append(0)
        elif token.startswith('im'):
            split_token = token[2:]
            if token not in positives:
                negated_event.append(split_token)
                neg_affix.append('im')
                hasnegaffix.append(1)
            else:
                negated_event.append('-')
                neg_affix.append('-')
                hasnegaffix.append(0)
        elif token.startswith('in'):
            split_token = token[2:]
            if token not in positives:
                negated_event.append(split_token)
                neg_affix.append('in')
                hasnegaffix.append(1)
            else:
                negated_event.append('-')
                neg_affix.append('-')
                hasnegaffix.append(0)
        elif token.startswith('non'):
            split_token = token[3:]
            if token not in positives:
                negated_event.append(split_token)
                neg_affix.append('non')
                hasnegaffix.append(1)
            else:
                negated_event.append('-')
                neg_affix.append('-')
                hasnegaffix.append(0)
        elif token.startswith('ir'):
            split_token = token[2:]
            if token not in positives:
                negated_event.append(split_token)
                neg_affix.append('ir')
                hasnegaffix.append(1)
            else:
                negated_event.append('-')
                neg_affix.append('-')
                hasnegaffix.append(0)
        elif token.endswith('less'):
            split_token = token[:-4]
            if token not in positives:
                negated_event.append(split_token)
                neg_affix.append('less')
                hasnegaffix.append(1)
            else:
                negated_event.append('-')
                neg_affix.append('-')
                hasnegaffix.append(0)
        elif token.endswith('lessly'):
            split_token = token[:-6]
            if token not in positives:
                negated_event.append(split_token)
                neg_affix.append('lessly')
                hasnegaffix.append(1)
            else:
                negated_event.append('-')
                neg_affix.append('-')
                hasnegaffix.append(0)
        elif token.endswith('lessness'):
            split_token = token[:-8]
            if token not in positives:
                negated_event.append(split_token)
                neg_affix.append('lessness')
                hasnegaffix.append(1)
            else:
                negated_event.append('-')
                neg_affix.append('-')
                hasnegaffix.append(0)
        else:
            negated_event.append('-')  # if token does not have any negational affixes:
            neg_affix.append('-')      # add dashes and 0 to appropriate lists
            hasnegaffix.append(0)

    # add affixal negation features to dataframe
    df['HasNegAffix'] = hasnegaffix
    df['Negated event'] = negated_event
    df['NegAffix'] = neg_affix

    # change filename and write data to .conll file
    outputfile = inputfile.replace('.txt','.features.conll')
    df.to_csv(outputfile, sep='\t', header=True, quotechar='|', index=False)

    outputfile2 = inputfile.replace('.txt', '.features.csv')
    df.to_csv(outputfile2, sep='\t', header=True, quotechar='|', index=False)

# function calls
add_features(inputfile_train)