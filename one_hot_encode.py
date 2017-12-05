import numpy as np

def one_hot_encode(data):
    # data will be in this formate -> [[3], [4], [9]....]
    encoded_data = []

    for i in range(len(data)):
        encoded = [0 for i in range(10)]

        # we put the value 1 which equals the index of the value of data
        encoded[data[i][0]] = 1

        encoded_data.append(encoded)

    return np.array(encoded_data)
