# imports
import pandas as pd
import nltk
import spacy

nlp = spacy.load("en_core_web_sm")
inputfile = "../Data/SEM-2012-SharedTask-CD-SCO-simple.v2/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt"

# open training data as pandas df + add column names
train_df = pd.read_csv(inputfile, sep="\t", header=None, names=["Book", "Sent nr", "Token nr", "Token", "Label"])

## Preprocessing: lowercasen

# put all our tokens into a list:
tokenlist = train_df['Token'].tolist()

# Code to extract previous token and next token:
pretoken = []
nexttoken = []

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

train_df['pretoken'] = pretoken
train_df['nexttoken'] = nexttoken

# Code for extracting POS tags with spacy:

d = []
for word in tokenlist:
    doc = nlp(word)
    poslist = []
    for item in doc:
        poslist.append(item.pos_)
    d.append(poslist)

newlist = []
for item in d:
    if len(item) > 1:
        newlist.append(item)
    else:
        newlist.append(item[0])

train_df["POS"] = newlist

d = []
print("processing words with spacy")
for word in tokenlist:
    doc = nlp(word)
    punctuation_list = []
    for item in doc:
        print(item.is_punct)
        if item.is_punct == True:
            punctuation_list.append("1")
        if item.is_punct == False:
            punctuation_list.append("0")
        else:
            print("Error: check word processing")
            break
    d.append(punctuation_list)
    print(word, punctuation_list)
print(d)


newlist = []
for item in d:
    if len(item) == 1:
        newlist.append(item)
    else:
        newlist.append(item[0])
train_df['Punctuation'] = newlist
# Code for extracting previous and next POS tag:



# Code for extracting the lemma, with spacy:

d = []
for word in tokenlist:
    doc = nlp(word)
    lemmalist = []
    for item in doc:
        lemmalist.append(item.lemma_)
    d.append(lemmalist)

newlist = []
for item in d:
    if len(item) > 1:
        newlist.append(item)
    else:
        newlist.append(item[0])
train_df['Lemma'] = newlist


# Code for extracting previous and next lemma:
lemmalist_2 = train_df['Lemma']
pre_lemma = []
next_lemma = []

i = 0
for lemma in lemmalist_2:
    if i == 0:
        pre_lemma.append('<BOD>')
        next_lemma.append(lemmalist_2[i+1])
    elif i == len(lemmalist_2) - 1:
        pre_lemma.append(lemmalist_2[i-1])
        next_lemma.append('<EOD>')
    else:
        pre_lemma.append(lemmalist_2[i-1])
        next_lemma.append(lemmalist_2[i+1])
    i += 1

train_df['pre_lemma'] = pre_lemma
train_df['next_lemma'] = next_lemma


print(train_df.head(20))

#print partial df to check if it went well
#print(train_df.head(10))




# convert pandas df to .conll file (uncomment when ready to use)
outputfile = "SEM_training_data.conll"
train_df.to_csv(f'c:/Users/desir/Desktop/text_mining/applied TM/{outputfile}', sep='\t', header=True, quotechar='|', index=False)
