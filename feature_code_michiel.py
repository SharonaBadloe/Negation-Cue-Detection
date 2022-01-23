# imports
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")
inputfile = "../Data/SEM-2012-SharedTask-CD-SCO-simple.v2/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt"

# open training data as pandas df + add column names
train_df = pd.read_csv(inputfile, sep="\t", header=None, names=["Book", "Sent nr", "Token nr", "Token", "Label"])

#code for extracting punctuation marks
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
train_df['Punctuation_spacy'] = newlist

#other option for extracting punctuation marks without the use of spacy

d = []
punct_list = string.punctuation
for token in df['Token']:
    if token in punct_list:
        d.append(1)
    else:
        d.append(0)
df['Punctuation_python'] = d


# code for classifying POS-tags
pos_cat_list = []
for tag in pos_list:
    if tag == "VERB" or tag == "AUX":
        pos_cat_list.append("VB")
    elif tag == "PRON":
        pos_cat_list.append("PRO")
    elif tag == "DET" or tag == "X" or tag == "NUM" or tag == "INTJ":
        pos_cat_list.append("OTH")
    elif tag == "ADV" or tag == "ADP" or tag == "SCONJ" or tag == "PART":
        pos_cat_list.append("ADVB")
    elif :
        pos_cat_list.append("VB")
    elif tag == "PROPN" or tag == "NOUN":
        pos_cat_list.append("NN")
    #if anything else, not seen in trainingdata comes forward:
    else:
        pos_cat_list.append("OTH")

train_df['POS_classified'] = pos_cat_list


# convert pandas df to .conll file (uncomment when ready to use)
outputfile = "SEM_training_data.conll"
train_df.to_csv(f'c:/Users/desir/Desktop/text_mining/applied TM/{outputfile}', sep='\t', header=True, quotechar='|', index=False)
