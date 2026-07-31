import numpy as np


def get_frequencies(array: np.ndarray, print_freqs: bool):

    elements, frequencies = np.unique(array, return_counts=True)

    rel_freq = frequencies / len(array)

    frequency_dict = dict(zip(elements, rel_freq))

    if print_freqs:
        for element, frequency in sorted(frequency_dict.items()):
            print(f"Element: {element!s:<10} | Frequency: {frequency}")

    return frequency_dict
