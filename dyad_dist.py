import os
from scipy.io import loadmat
import numpy as np
import makefilter
from scipy.signal import sosfiltfilt, hilbert
from scipy.signal import savgol_filter
import pickle
from itertools import combinations

def list_data_folders(directory_path,pattern):
    """
    Lists all folders in the given directory that start with pattern.
    
    Parameters:
    - directory_path: A string representing the path to the directory to search in.

    - pattern: A string representing the pattern at the beginning of the folders' name.
    
    Returns:
    - A list of folder names that meet the criteria.
    """
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return []

    # Get all items in the directory
    all_items = os.listdir(directory_path)

    # Filter for directories that start with pattern
    folders_starting_with_pattern = [item for item in all_items
                                if os.path.isdir(os.path.join(directory_path, item)) and item.startswith(pattern)]

    return folders_starting_with_pattern

def list_files_in_directory(directory_path):
    """
    Lists all the files in the specified directory.
    
    Parameters:
    - directory_path: A string representing the path to the directory to search in.
    
    Returns:
    - A list of file names contained in the directory.
    """
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return []

    # Get all items in the directory
    all_items = os.listdir(directory_path)

    # Filter out only files
    files_only = [item for item in all_items if os.path.isfile(os.path.join(directory_path, item))]

    return files_only

def open_mat_file(file_path):
    """
    Opens a .mat file and returns its contents.
    
    Parameters:
    - file_path: A string representing the path to the .mat file.
    
    Returns:
    - A dictionary containing variables loaded from the .mat file.
    """
    # Load the .mat file
    data = loadmat(file_path)
    
    return data

def segment_time_series(data, segment_duration, overlap_duration, sampling_rate):
    """
    Segments each time series in a 2D NumPy array into smaller time series segments 
    based on specified duration, overlap, and sampling rate.

    Parameters:
    - data (numpy.ndarray): A 2D NumPy array containing the time series data. Each 
    column represents a single time series.
    - segment_duration (float): The duration of each segment in seconds. Determines
     the length of each segment generated.
    - overlap_duration (float): The duration of overlap between consecutive segments
     in seconds. This specifies how much each segment should overlap with the next.
    - sampling_rate (int): The number of samples per second in the time series data.
     This is used to calculate the number of samples per segment and the overlap in 
     samples.

    Returns:
    - numpy.ndarray: A 3D NumPy array where the 1st dimension represents a segmented portion 
    of the original time series. The 2nd dimension in this array equals the number 
    of samples per segment, determined by the `segment_duration` and `sampling_rate`.
    """
    # Calculate parameters
    samples_per_segment = int(segment_duration * sampling_rate)
    overlap_samples = int(overlap_duration * sampling_rate)
    step_size = samples_per_segment - overlap_samples
    
    segments = []
    len_data,_ = data.shape
    for start in range(0, len_data - samples_per_segment + 1, step_size):
        segment = data[start:start+samples_per_segment,:]
        segments.append(segment)
            
    return np.array(segments)

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

def corr_matrix_stack(corr_matrix_dic):
    """
    Stacks correlation matrices along axis 0 and tracks their sizes and identifiers.

    This function takes a dictionary of correlation matrices (2D NumPy arrays) and
    stacks these matrices vertically. It also compiles a list of the sizes of these
    matrices along axis 0, alongside their identifiers.

    Parameters:
    - corr_matrix_dic (dict): A dictionary where the keys are identifiers (e.g., file names)
      and the values are correlation matrices (2D NumPy arrays). Each matrix is assumed to
      have the same number of columns but can vary in the number of rows.

    Returns:
    - numpy.ndarray: A single 2D NumPy array resulting from stacking all the input matrices
      along axis 0.
    - list of tuples: Each tuple contains an identifier (key from the input dictionary) and
      an integer representing the size (number of rows) of the corresponding matrix before
      stacking. This list maintains the order in which matrices were stacked.
    """
    
    # Initialize a list to keep track of each matrix's identifier and its number of rows
    size_matrices = []
    
    # Initialize a list to hold all matrices for concatenation
    matrix_list = []
    
    # Iterate over the dictionary items
    for file, matrix in corr_matrix_dic.items():
        # Append each matrix to the list for later concatenation
        matrix_list.append(matrix)
        # Append a tuple of the matrix's identifier and its number of rows to the tracking list
        size_matrices.append((file, matrix.shape[0]))
    
    # Concatenate all matrices vertically
    stacked_array = np.concatenate(matrix_list, axis=0)
    
    # Return the stacked array and the list of identifiers with their corresponding matrix sizes
    return stacked_array, size_matrices

def corr_dist(A,B):
    from scipy.linalg import norm
    D = np.transpose(np.conj(A))@B
    dist = np.real(np.log(1.0/(np.trace(D)/(norm(A)*norm(B)))))
    return dist

def sort_key(filename):
    trial_part = int(filename.split('_')[-1].split('.')[0])  # Extract the trial number
    return trial_part

directory_path = '/data/cortical_source_data/'  # Replace this with the path to your directory
folders = list_data_folders(directory_path,'20')

#Butterworth filters.
butt_filter1,butt_w,butt_h = makefilter.makefiltersos(2000,50,60)
window_len = 2.0
overlap = 0.5
overlap_len = overlap*window_len
butt_filter2,butt_w,butt_h = makefilter.makefiltersos(2000,1.0/window_len,0.5/window_len)

# folders = ['20220713','20220721',
#            '20220804','20220808',
#            '20220810','20220811',
#            '20220815','20220816',
#            '20221003','2022100401',
#            '2022100402','20221005']

for folder in folders:
    print(folder)
    files_dic = {}
#    for folder in folders:
#    folder_dic = {}
    directory_path = '/data/cortical_source_data/' + folder
    files = list_files_in_directory(directory_path)

    for filename in files:
        subject, rest = filename.split('_', 1)
        if subject not in files_dic:
            files_dic[subject] = {}
        file = folder + '/' + filename
        files_dic[subject][filename] = directory_path + '/' + filename
    print('files listed.')

    subject_matrices = {}
    for subject in ['subj1','subj2']:
        print(subject) 
        corr_matrices = {}
        subject_files = list(files_dic[subject].keys())
        subject_files = sorted(subject_files,key=sort_key)

        for file in subject_files:
            print(file)
            path = files_dic[subject][file]
            filename = folder + '/' + file
            # print(path)

            #Import the data from a sample file.
            data = open_mat_file(path)

            #Get signal, filter and downsample.
            signal = data['agr_source_data']
            filtered_signal = sosfiltfilt(butt_filter1, signal, axis=0)

            filtered_signal = sosfiltfilt(butt_filter2, filtered_signal, axis=0)
            downsampled_signal = filtered_signal[::10,:]

            analytic_signal = hilbert(downsampled_signal,axis=0)
            sg = int(np.floor(100/(25))*2+1)
            ts1 = savgol_filter(np.real(analytic_signal),sg,1,axis = 0,mode = 'interp')
            ts2 = savgol_filter(np.imag(analytic_signal),sg,1,axis = 0,mode = 'interp')
            ts3 = ts1+1j*ts2
            time_windows = segment_time_series(ts3, window_len, overlap_len, 200)

            corr_matrix = []
            for window in time_windows:
                corr_matrix.append(np.corrcoef(window, rowvar=False))
            corr_matrix = np.array(corr_matrix)
            corr_matrices[filename] = corr_matrix
        print('correlation matrices calculated.')

        matrix,sizes = corr_matrix_stack(corr_matrices)
        dist_matrix = np.zeros((len(matrix),len(matrix)))
        for i,j in combinations(range(len(matrix)),2):
            print(i)
            matrix1 = matrix[i,:,:]
            matrix2 = matrix[j,:,:]
            dist_matrix[i,j] = corr_dist(matrix1,matrix2)
            dist_matrix[j,i] = dist_matrix[i,j]
        
        print('distance matrices calculated.')
        subject_matrices[subject] = {'distances': dist_matrix, 'sizes':sizes}
    
    filename_save = '/data/Italo/correlation_distances/dyad_' + folder + '_distances.pkl'
    print(filename_save)
    save_to_pickle(subject_matrices, filename_save)
