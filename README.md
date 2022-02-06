# Applied Text Mining Methods Project
Project code for Text Mining Methods course - Vrije Universiteit, Amsterdam The Netherlands.
Directory created by Sharona Badloe, Desiree Gerritsen, Jingyue Zhang and Michiel van Nederpelt. 

# Project Description:
This project focusses on negation cue detection. A task where negation cues (e.g., words or affixes) are detected based on our own built classifier. 
For more information about related work and our features, see the report.

**REQUIREMENTS**

- Python 3.6+ (recommended: Python 3.7)
- SpaCy 3.1.3
- NLTK 3.6.3

Downloads: 
- Spacy library: https://spacy.io/usage
- NLTK library: https://www.nltk.org/_modules/nltk/downloader.html

Modules:
- matplotlib
- pandas
- collections 

# How to run

**The directory contains the following files and folders:**
- annotations
- results
- data
- README.md
- basic_statistics.py
- feature_extraction.py
- bioscope_feature_extraction.py
- lexicon_baseline.py
- SVM_feature_ablation.py
- error_analysis.ipynb

**basic_statistics.py: (runtime around 2 minutes)**
This script can be run in your preferred IDE, or from the terminal by typing 'python basic_statistics.py'.
The following three files are outputted in the results directory:
- Figure 1: The distribution of POS-tags for negations in the training and development dataset (negation_pos_distribution.png)
- Figure 2: The distribution of POS-tags for the training dataset (count_pos_train.png)
- Figure 3: The distribution of POS-tags for the development dataset (count_pos_dev.png)

**feature_extraction.py: (runtime around 15 minutes)**
This script can be run in your preferred IDE, or from the terminal by typing 'python feature_extraction.py'.
The following four files are outputted in the results directory: 

- training data with added features in .csv format
- development data with added features in .csv format
- training data with added features in .conll format
- development data with added features in .conll format

#The .csv documents are for easy viewing of the features in a nice table format. The .conll documents will be utilized in our classification process.

**bioscope_feature_extraction.py: (runtime around 15 minutes)**
This script performs the same actions and outputs the same results as the feature_extraction.py file, but it contains some extra preprocessing steps specific to the bioscope data. 

**lexicon_baseline.py (runtime around 5 minutes)**
This file contains the code for creating a negation lexicon and using it in a rule-based system. It will output the length of the 
negation lexicon, and two classification reports. One report for the dev set, and one for the training set.

**SVM_feature_ablation.py: (runtime around 2 minutes)**
This script can be used to showcase the performance of the SVM classifier, for both all features and selected features.
Make sure to store the data in the right folders, and check the path.
The script will provide you with the evaluation metrics for both all features and selected features (which can easily be altered).

**error_analysis.ipynb**
This notebook extracts and displays some statistics on the amount of errors and correctly predicted tokens. It also extracts sentences, sentence numbers and labels for easy analysis. This notebook has been utilized in our partly manual, partly automated process of error analysis.

