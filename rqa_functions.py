import numpy as np

def build_rp(cluster_sequence):
    # Determine the length of the sequence
    sequence_length = len(cluster_sequence)

    # Initialize a square matrix to represent the recurrence plot
    recurrence_matrix = np.zeros((sequence_length, sequence_length))

    # Fill the matrix
    for i in range(sequence_length):
        for j in range(sequence_length):
            if cluster_sequence[i] == cluster_sequence[j]:
                recurrence_matrix[i, j] = 1

    return recurrence_matrix

def calculate_rr(recurrence_matrix):
    """
    Calculate the Recurrence Rate (RR) from a recurrence matrix.
    """
    total_points = recurrence_matrix.size
    recurrent_points = recurrence_matrix.sum()
    rr = recurrent_points / total_points
    return rr

def find_lines(rp, min_len=2, direction='diagonal'):
    """
    Identify lines in a recurrence plot.

    Parameters:
    - rp: numpy array, the recurrence plot.
    - min_len: int, minimum length of line to be considered.
    - direction: str, 'diagonal' or 'vertical' to specify line orientation.

    Returns:
    - lines: list of ints, lengths of identified lines.
    """
    n = rp.shape[0]
    lines = []
    if direction == 'diagonal':
        for i in range(n):
            for j in range(i+1, n):
                length = 0
                while i + length < n and j + length < n and rp[i + length, j + length]:
                    length += 1
                if length >= min_len:
                    lines.append(length)
                    # Skip to the end of this line to avoid recounting
                    i += length - 1
    elif direction == 'vertical':
        for j in range(n):
            length = 0
            for i in range(n):
                if rp[i, j]:
                    length += 1
                else:
                    if length >= min_len:
                        lines.append(length)
                    length = 0
            # Check if the last sequence in the column is a line
            if length >= min_len:
                lines.append(length)
    return lines

def determinism(rp, min_len=2):
    """
    Calculate the determinism of a recurrence plot.
    """
    diagonal_lines = find_lines(rp, min_len=min_len, direction='diagonal')
    det = sum(l for l in diagonal_lines) / rp.sum()
    return det

def avg_line_length(rp, min_len=2, direction='diagonal'):
    """
    Calculate the average line length of either diagonal or vertical lines.
    """
    lines = find_lines(rp, min_len=min_len, direction=direction)
    if lines:
        return np.mean(lines)
    return 0

def max_line_length(rp, min_len=2, direction='diagonal'):
    """
    Calculate the maximum line length of either diagonal or vertical lines.
    """
    lines = find_lines(rp, min_len=min_len, direction=direction)
    if lines:
        return np.max(lines)
    return 0

def laminarity(rp, min_len=2):
    """
    Calculate the laminarity of a recurrence plot.
    """
    vertical_lines = find_lines(rp, min_len=min_len, direction='vertical')
    lam = sum(l for l in vertical_lines) / rp.sum()
    return lam
