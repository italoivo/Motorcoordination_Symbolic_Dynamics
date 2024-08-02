import numpy as np
from scipy.io import loadmat
import pickle
import pandas as pd

def load_pickle_file(filename):
    """
    Load a pickle file.

    Parameters:
    - filename (str): The path to the pickle file to be loaded.

    Returns:
    - The Python object loaded from the pickle file.
    """
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except EOFError:
        print(f"Error: The file '{filename}' may be corrupted or empty.")
    except pickle.UnpicklingError:
        print(f"Error: The file '{filename}' could not be unpickled. It may not be a valid pickle file or may be corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def save_to_pickle(dict_obj, file_path):
    """
    Saves a given dictionary to a pickle file.

    Parameters:
    - dict_obj (dict): The dictionary to be saved.
    - file_path (str): The path to the file where the dictionary will be saved.

    Returns:
    - None
    """
    with open(file_path, 'wb') as file:
        # Serialize the dictionary and save it to the file
        pickle.dump(dict_obj, file)
    print(f"Data successfully saved to {file_path}.")

def find_indices_to_eliminate(subj1, subj2):
    """
    Calculate the indices to be eliminated based on the differences in trial data points
    for two subjects, ensuring that only the necessary data points are removed to align their sizes.

    Parameters:
    - list_indices_subj1: Numpy array of trial sizes for subject 1
    - list_indices_subj2: Numpy array of trial sizes for subject 2

    Returns:
    - index_to_eliminate_subj1: Indices to eliminate from subject 1 to align with subject 2
    - index_to_eliminate_subj2: Indices to eliminate from subject 2 to align with subject 1
    """

    list_indices_subj1 = np.array([i[1] for i in subj1])
    list_indices_subj2 = np.array([i[1] for i in subj2])

    cumsum_subj1 = [sum([x[1] for x in subj1[:i+1]]) for i in range(len(subj1))]
    cumsum_subj2 = [sum([x[1] for x in subj2[:i+1]]) for i in range(len(subj2))]

    index_differences_sub1 = list_indices_subj1 - list_indices_subj2
    index_differences_sub2 = list_indices_subj2 - list_indices_subj1

    index_to_eliminate_subj1 = []
    for i,n_points in enumerate(index_differences_sub1):
        if n_points>0:
            indexes = [j for j in range(cumsum_subj1[i]-index_differences_sub1[i],cumsum_subj1[i])]
            index_to_eliminate_subj1.extend(indexes)

    index_to_eliminate_subj2 = []
    for i,n_points in enumerate(index_differences_sub2):
        if n_points>0:
            indexes = [j for j in range(cumsum_subj2[i]-index_differences_sub2[i],cumsum_subj2[i])]
            index_to_eliminate_subj2.extend(indexes)

    return index_to_eliminate_subj1[::-1], index_to_eliminate_subj2[::-1]

def session_data_loading(file_path):

    session = (file_path.split('/')[-1]).split('_')[1]
    with open(file_path, 'rb') as file:
        # Load the object from the pickle file
        data = pickle.load(file)

    subj1 = data['subj1']['sizes']
    subj2 = data['subj2']['sizes']

    index_to_eliminate_subj1, index_to_eliminate_subj2 = find_indices_to_eliminate(subj1, subj2)

    file_order_size = []
    for i in range(len(subj1)):
        file_sub1,len_1 = subj1[i]
        file_sub2,len_2 = subj2[i]
        if len_1 < len_2:
            file_order_size.append((file_sub1,file_sub2,len_1))
        else:
            file_order_size.append((file_sub1,file_sub2,len_2))

    mat1 = data['subj1']['distances']
    for index in index_to_eliminate_subj1:
        mat1 = np.delete(np.delete(mat1, index, axis=0), index, axis=1)

    mat2 = data['subj2']['distances']
    for index in index_to_eliminate_subj2:
        mat2 = np.delete(np.delete(mat2, index, axis=0), index, axis=1)

    trial_len = [i[2] for i in file_order_size]
    start_points = list(np.cumsum(trial_len))
    end_points = [i-1 for i in start_points]
    start_points.insert(0,0)
    start_points.pop(-1)
    #print(start_points)
    #print(end_points)
    start_stop = list(zip(start_points,end_points))
    #print(start_stop)

    condition_dictionary = {1: 'Uncoupled', 2: '1_lead', 3: '2_lead', 4: 'Mutual'}
    type_dictionary = {1: 'Synchronization', 2: 'Syncopation'}

    # Initialize an empty list to store each row's data as a dictionary
    data = []

    for i, entry in enumerate(file_order_size):
        session = entry[0].split('/')[0]
        trial = entry[0].split('_')[2][:-4]
        length = entry[2]
        start, stop = start_stop[i]
        filename = '/data/Italo/finger_tapping_behavioral_data/clean_' + str(session) + '_bpchan.mat'
        beh_data = loadmat(filename)
        conditions = list(beh_data['conditions'][0])
        condition = condition_dictionary[conditions[int(trial)-1]]
        session_type = type_dictionary[beh_data['session'][0][0]]

        # Instead of printing, store the data in a dictionary
        row_data = {
            'session': session,
            'session_type': session_type,
            'condition': condition,
            'trial': trial,
            'start': start,
            'stop': stop
        }

        # Append the dictionary to the list
        data.append(row_data)

    # Convert the list of dictionaries to a pandas DataFrame
    metadata = pd.DataFrame(data)
    session_data = {'Subject 1': mat1,
                    'Subject 2': mat2,
                    'Metadata': metadata,
                    'Session Type': session_type}
    return session, session_data

def bootstrap_p_value(sample1, sample2, num_bootstrap=10000, stat='mean'):
    """
    Estimate the p-value using a bootstrap method for the difference between two samples.
    
    Parameters:
    - sample1, sample2: numpy.ndarray, the two samples for comparison.
    - num_bootstrap: int, the number of bootstrap resamples to perform.
    - stat: str, the statistic to compare ('mean' or 'median').
    
    Returns:
    - p_value: float, the estimated p-value for the difference between the two samples.
    """
    # Calculate the observed statistic
    if stat == 'mean':
        observed_stat = np.mean(sample1) - np.mean(sample2)
    elif stat == 'variance':
        observed_stat = np.var(sample1) - np.var(sample2)
    else:
        raise ValueError("stat parameter must be 'mean' or 'variance'")
    
    # Combine the samples
    combined_samples = np.concatenate([sample1, sample2])
    
    # Initialize a list to store the bootstrap statistics
    bootstrap_stats = []
    
    for _ in range(num_bootstrap):
        # Resample with replacement
        resample1 = np.random.choice(combined_samples, size=len(sample1), replace=True)
        resample2 = np.random.choice(combined_samples, size=len(sample2), replace=True)
        
        # Calculate the statistic for the bootstrap samples
        if stat == 'mean':
            bootstrap_stat = np.mean(resample1) - np.mean(resample2)
        elif stat == 'variance':
            bootstrap_stat = np.var(resample1) - np.var(resample2)
        
        bootstrap_stats.append(bootstrap_stat)
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate the p-value
    if observed_stat > 0:
        p_value = np.sum(bootstrap_stats >= observed_stat) / num_bootstrap
    else:
        p_value = np.sum(bootstrap_stats <= observed_stat) / num_bootstrap
    
    return observed_stat,p_value

def transform_tuples_to_symbols(tuple_sequence):
    """
    Transforms a sequence of tuples into a sequence of unique symbols (integer numbers).
    
    Parameters:
    - tuple_sequence: A sequence (e.g., list) of tuples.
    
    Returns:
    - A list of integers representing the sequence of symbols.
    """
    # Step 1: Create a mapping from each unique tuple to a unique integer
    unique_tuples = set(tuple_sequence)  # Find all unique tuples
    tuple_to_symbol_map = {t: i for i, t in enumerate(unique_tuples)}
    
    # Step 2: Transform the original sequence of tuples using the map
    symbol_sequence = [tuple_to_symbol_map[t] for t in tuple_sequence]
    
    return symbol_sequence