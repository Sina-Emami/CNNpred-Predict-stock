import random
import numpy as np


def datagen(data, seq_len, batch_size, targetcol, kind, TRAIN_TEST_CUTOFF, TRAIN_VALID_RATIO):
    """
    It takes a dictionary of dataframes, and returns a generator that produces batches of data for
    training
    
    :param data: a dictionary of dataframes, each of which contains the data for one stock
    :param seq_len: the length of the sequence
    :param batch_size: the number of samples in a batch
    :param targetcol: the column name of the target variable
    :param kind: 'train' or 'valid'
    """
    "As a generator to produce samples for Keras model"
    batch = []
    while True:
        # Pick one dataframe from the pool
        key = random.choice(list(data.keys()))
        df = data[key]
        input_cols = [c for c in df.columns if c != targetcol]
        index = df.index[df.index < TRAIN_TEST_CUTOFF]
        split = int(len(index) * TRAIN_VALID_RATIO)
        assert split > seq_len, "Training data too small for sequence length {}".format(seq_len)
        if kind == 'train':
            index = index[:split]   # range for the training set
        elif kind == 'valid':
            index = index[split:]   # range for the validation set
        else:
            raise NotImplementedError
        # Pick one position, then clip a sequence length
        while True:
            t = random.choice(index)     # pick one time step
            n = (df.index == t).argmax() # find its position in the dataframe
            if n-seq_len+1 < 0:
                continue # this sample is not enough for one sequence length
            frame = df.iloc[n-seq_len+1:n+1]
            batch.append([frame[input_cols].values, df.loc[t, targetcol]])
            break
        # if we get enough for a batch, dispatch
        if len(batch) == batch_size:
            X, y = zip(*batch)
            X, y = np.expand_dims(np.array(X), 3), np.array(y)
            yield X, y
            batch = []


def testgen(data, seq_len, targetcol, TRAIN_TEST_CUTOFF):
    """
    > For each key in the dictionary, find the first index of the test sample, then for each index in
    the test sample, create a sequence of length `seq_len` and append it to the batch
    
    :param data: a dictionary of dataframes, one for each stock
    :param seq_len: the length of the sequence of data to be fed into the model
    :param targetcol: the column to predict
    :return: A tuple of two arrays. The first array is the input data, and the second array is the
    target data.
    """
    "Return array of all test samples"
    batch = []
    for key, df in data.items():
        input_cols = [c for c in df.columns if c != targetcol]
        # find the start of test sample
        t = df.index[df.index >= TRAIN_TEST_CUTOFF][0]
        n = (df.index == t).argmax()
        for i in range(n+1, len(df)+1):
            frame = df.iloc[i-seq_len:i]
            batch.append([frame[input_cols].values, frame[targetcol][-1]])
    X, y = zip(*batch)
    return np.expand_dims(np.array(X),3), np.array(y)