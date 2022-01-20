#Import packages
import pandas as pd
import spacy

# count the number of tokens
def count_tokens(conllfile):
    '''Function that counts for number of tokens.
    
    :param conllfile: path to the (preprocessed) conll file
    :type conllfile: string
    
    
    :return features: number of tokens.
    '''  
    #create a list to capture all tokens
    tokens = []
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t')
    for row in csvreader:
        token_value = row[3]
        tokens.append(token_value)
            
    return len(tokens) #this number including headerline, so the result should be subtracted by 1


#--------------
#count for the number of negation cues in total and in 'B-NEG' and 'I-NEG'
def count_negation_cues(conllfile):
    '''Function that counts for number of negation cues of each category.
    
    :param conllfile: path to the (preprocessed) conll file
    :type conllfile: string
    
    
    :return features: number of ngation cue.
    ''' 
    # create a list to collect all negation cues
    negation_cues = []
    cue1=[] # cue1 collects 'B-NEG'
    cue2=[] # cue2 collects 'I-NEG'
    
    conllinput = open(conllfile, 'r')
    csvreader = csv.reader(conllinput, delimiter='\t')
    for row in csvreader:
        if row[4] != 'O':
            negation = row[4]
            negation_cues.append(negation)
   
    for cues in negation_cues:
        if cues=='B-NEG':
            cue1.append(cues)
        if cues=='I-NEG':
            cue2.append(cues)

    #number of len(negation_cues) including headerline, so the result should be subtracted by 
    return len(negation_cues),len(cue1),len(cue2) 


#--------------
