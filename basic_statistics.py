from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


# inputfiles 
inputfile_train = 'data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.features.conll'
inputfile_dev = 'data/SEM-2012-SharedTask-CD-SCO-dev-simple.v2.features.conll'

# outputfiles
outputfile_train_pos = 'results/count_pos_train.png'
outputfile_dev_pos = 'results/count_pos_dev.png'
outputfile_negation_pos = 'results/negation_pos_distribution.png'


def count_tokens_dataset(path_to_file):
    """
    A function to count the distribution of POS-tags for the given dataset:
    It requires a file that contains a 'POS' column, generated with NLTK.
    Store the outcome of this function in a variable, which can be used for the plot_distribution_pos_alltokens function. 
    """
    
    dataframe = pd.read_csv(path_to_file, sep="\t")
    
    pos = dataframe['POS']

    punct = []
    postags = []
    punctlist = '''!()-[]{};:'"\,<>./?@#$%^&*``''_~'''
    for postag in pos:
        #print(postag)
        if postag in punctlist:
            punct.append(postag)
        else:
            postags.append(postag)

    pos_counter = Counter(postags)
    #print(pos_counter)
    return pos_counter

def plot_distribution_pos_alltokens(counter_object, output_file):
    """
    A function that plots the distribution of POS-tags for all tokens of a given dataset.
    :param counter_object: a variable that contains the counter output.
    :param output_file: filename for the saved plot.
    :returns: saved plots in results directory
    """
    
    pos_counter = counter_object
    
    plt.figure(figsize=(15,15))
    plt.bar(pos_counter.keys(), pos_counter.values())
    plt.savefig(output_file)

    
    
def count_negations_dataset(path_to_dev_file, path_to_train_file, output_file_name):
    """
    A function that counts the distribution of POS-tags for the negations of a given datasets.
    It requires both files to contain a 'Label' column and a 'POS' column.
    """  
    
    dev_data = pd.read_csv(path_to_dev_file, sep="\t")
    train_data = pd.read_csv(path_to_train_file, sep="\t")
    
    # Prepare the development negations:
    df_bneg_dev = dev_data.loc[dev_data['Label'] == 'B-NEG', 'POS']
    #print(df_bneg)
    df_ineg_dev = dev_data.loc[dev_data['Label'] == 'I-NEG', 'POS']
    #print(df_ineg)

    # Merge the two frames together: 
    frames_dev = [df_bneg_dev, df_ineg_dev]
    frames_new_dev = pd.concat(frames_dev)
    #print(frames_new_dev)

    # Prepare the training negations:
    df_bneg_train = train_data.loc[train_data['Label'] == 'B-NEG', 'POS']
    #print(df_bneg)
    df_ineg_train = train_data.loc[train_data['Label'] == 'I-NEG', 'POS']
    #print(df_ineg)
    
    # Merge the two frames together: 
    frames_train = [df_bneg_train, df_ineg_train]
    frames_new_train = pd.concat(frames_train)
    #print(frames_new_train)
    
    negation_distribution_train = frames_new_train
    negation_distribution_dev = frames_new_dev

    # Count the negations:
    negation_counter_train = Counter(negation_distribution_train)
    negation_counter_dev = Counter(negation_distribution_dev)

#     print(negation_counter_train)
#     print(negation_counter_dev)

    
    # Create a pandas dataframe from the training counter dict output:
    dataframe_train = pd.DataFrame.from_dict(negation_counter_train, orient='index').reset_index()
    # Rename the column names:
    dataframe_train_new = dataframe_train.rename(columns={'index':'POS', 0:'Training'})


    # Create a pandas dataframe from the development counter dict output:
    dataframe_dev = pd.DataFrame.from_dict(negation_counter_dev, orient='index').reset_index()
    # Rename the column names:
    dataframe_dev_new =dataframe_dev.rename(columns={'index':'POS', 0:'Development'})

    
    # Merge the two dataframes on the POS label:
    merged_dataframes = dataframe_train_new.merge(dataframe_dev_new, how='left', left_on='POS', right_on='POS')

    # Optional: Test to see how if works.
    #print(merged_dataframes)
    
    # Create a plot with data of both datasets:
    plot_negations = merged_dataframes.plot.bar(x='POS', figsize=(10,10))
    fig = plot_negations.get_figure()
    
    # Save the plot:
    fig.savefig(output_file_name)
    

# function calls:

count_train = count_tokens_dataset(inputfile_train)
count_dev = count_tokens_dataset(inputfile_dev)
plot_distribution_pos_alltokens(count_train, outputfile_train_pos)
plot_distribution_pos_alltokens(count_dev, outputfile_dev_pos)

count_negations_dataset(inputfile_dev, inputfile_train, outputfile_negation_pos)

