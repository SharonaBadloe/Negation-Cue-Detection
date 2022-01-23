from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

def count_tokens_dataset(path_to_file):
    """
    A function to count the distribution of POS-tags for the given dataset:
    It requires a file that contains a POS column, generated with NLTK.
    Store the outcome of this function in a variable, which can be used for the plot_distribution_pos_alltokens function. 
    """
    
    dataframe = path_to_file
    
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
    Input 1 = a variable that contains the counter output.
    Input 2 = filename for the saved plot.
    
    """
    
    pos_counter = counter_object
    
    plt.figure(figsize=(15,15))
    plt.bar(pos_counter.keys(), pos_counter.values())
    plt.savefig(f'../{output_file}.png')

    
    
def count_negations_dataset(path_to_dev_file, path_to_train_file, output_file_name):
    """
    A function that counts the distribution of POS-tags for the given datasets.
    It requires both files that contain a 'Label' column and a 'POS' column.
    
    """  
    
    dev_data = path_to_dev_file
    train_data = path_to_train_file
    
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
    df_bneg_train = training_data.loc[training_data['Label'] == 'B-NEG', 'POS']
    #print(df_bneg)
    df_ineg_train = training_data.loc[training_data['Label'] == 'I-NEG', 'POS']
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
    joejoe_train = pd.DataFrame.from_dict(negation_counter_train, orient='index').reset_index()
    # Rename the column names:
    joejoe_train_new = joejoe_train.rename(columns={'index':'POS', 0:'Training'})
    #print(joejoe)

    # Create a pandas dataframe from the development counter dict output:
    joejoe_dev = pd.DataFrame.from_dict(negation_counter_dev, orient='index').reset_index()
    # Rename the column names:
    joejoe_dev_new = joejoe_dev.rename(columns={'index':'POS', 0:'Development'})

    
    # Merge the two dataframes on the POS label:
    merged_dataframes = joejoe_train_new.merge(joejoe_dev_new, how='left', left_on='POS', right_on='POS')

    # Optional: Test to see how if works.
    #print(merged_dataframes)
    
    # Create a plot with data of both datasets:
    plot_negations = merged_dataframes.plot.bar(x='POS', figsize=(10,10))
    fig = plot_negations.get_figure()
    
    # Save the plot:
    fig.savefig(f'../{output_file_name}.png')
