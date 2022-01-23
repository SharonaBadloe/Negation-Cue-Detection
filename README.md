# Applied Text Mining Methods Project
Project code for Text Mining Methods course - Vrije Universiteit, Amsterdam The Netherlands.
Directory created by Sharona Badloe, Desiree Gerritsen, Jingyue Zhang and Michiel van Nederpelt. 

# Project Description:
This project focusses on negation cue detection. A task where negation cues (e.g., words or affixes) are detected based on our own build classifier. 
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

**HOW TO RUN**

The directory contains the following files and folders: 
- annotations
- data
- README.md
- basic_statistics.py
- feature_extraction.py

basic_statistics.py: (runtime around 2 minutes)
This script can be run in your preferred IDE, or from the terminal by typing 'python basic_statistics.py'.
The following three files are outputted in the same directory as the script:
- Figure 1: The distribution of POS-tags for negations in the training and development dataset.
- Figure 2: The distribution of POS-tags for the training dataset.
- Figure 3: The distribution of POS-tags for the development dataset.

feature_extraction.py: (runtime around 15 minutes)
This script can be run in your preferred IDE, or from the terminal by typing 'python feature_extraction.py'.
The following four files are outputted in the same directory as the script: 

- training data with added features in .csv format
- development data with added features in .csv format
- training data with added features in .conll format
- development data with added features in .conll format

The .csv documents are for easy viewing of the features in a nice table format. 
The .conll documents will be utilized in our classification process.
