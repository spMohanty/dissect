'''
Created on Sep 26, 2012

@author: Georgiana Dinu, Pham The Nghia
'''
from numpy import  dtype, float32, fromstring
# from composes.semantic_space.space import Space

import composes.semantic_space.space



def list2dict(list_):
    return_dict = {}

    for idx, word in enumerate(list_):
        if word in return_dict:
            raise ValueError("duplicate string found in list: %s" % (word))
        return_dict[word] = idx

    return return_dict

def add_items_to_dict(dict_, list_):

    no_els = len(dict_)
    for idx, el in enumerate(list_):
        if el in dict_:
            raise ValueError("Found duplicate keys when appending elements to\
                            dictionary.")
        dict_[el] = no_els + idx
    return dict_

def assert_dict_match_list(dict_, list_):

    match_err = ValueError("expected matching dictionary and list structures.")

    if not len(list_) == len(dict_):
        raise match_err
    for (k, v) in dict_.iteritems():
        if not list_[v] == k:
            raise match_err


def assert_shape_consistent(matrix_, id2row, id2column, row2id, column2id):

    no_rows = matrix_.mat.shape[0]
    no_cols = matrix_.mat.shape[1]

    has_column_maps = column2id or id2column

    if not no_rows == len(id2row) or not no_rows == len(row2id):
        raise ValueError("expected consistent shapes: %d %d %d"
                         % (no_rows, len(id2row), len(row2id)))

    if (has_column_maps and
        (not no_cols == len(id2column) or not no_cols == len(column2id))):
        raise ValueError("expected consistent shapes: %d %d %d"
                         % (no_cols, len(id2column), len(column2id)))


def load_word2vec_bnary_as_space(filename):
    #!/usr/bin/env python
    f=open(filename,"rb")


    ##Read word_size and hidden_layer1 size
    header = f.readline()
    print header
    vocab_size, layer1_size = map(int, header.split()) 
    # print vocab_size, layer1_size
    ##Assuming only binary File

    data_store = {}

    # A Linear Transformation to make all the dimensions of all the feature vectors is necessary 
    # As DISSECT used NonNegative Matrix Factorization and hence doesnot support non negative values 
    #
    # And, a linear transformation is the best bet, as we do not want to loose any of the statistical properties of the distribution
    # by passing them through a funny function.

    # Do we will do two iterations in this pre-processing stage just to leave the statistical distribution of the params unaffectedlayer1_min = {} #Holds the min for all dimensions (needed to translate the vector space to sit well with DISSECT as it doesnot support negative params)
    layer1_min = {} # Dictionary to help computer bounds of all dimensions
    layer1_is_initialized = {} #Flag dict to intiialize on the first go
    for k in range(layer1_size): layer1_is_initialized[str(k)] = False;
    binary_length = dtype(float32).itemsize * layer1_size

    for line in xrange(vocab_size):
        # mixed text and binary: read text first, then binary
        word = []
        while True:
            ch = f.read(1)
            if ch == b' ':
                break
            if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                word.append(ch)

        word = "".join(word) # Compose string from char array
        feature_vector = fromstring(f.read(binary_length), dtype=float32)
        for k in range(len(feature_vector)):        
            if not layer1_is_initialized[str(k)]:
                layer1_min[str(k)] = feature_vector[k]
                layer1_is_initialized[str(k)] = True
            else:
                if( feature_vector[k] < layer1_min[str(k)]):
                    layer1_min[str(k)] = feature_vector[k]
        data_store[word] = feature_vector



    ## Bounds of all the data points assessed after first pass
    f.close()
    f = open("/tmp/temp_word2vec_model.dm", "w")
    for word in data_store.keys():
        temp_feature = data_store[word]
        ##Translate so that all values are positive
        for k in range(len(temp_feature)):
            temp_feature[k] -= layer1_min[str(k)]

        f.write(word+" ")
        for k in temp_feature:
            f.write(str(k)+" ")
        f.write("\n")

    f.close()

    return (True,composes.semantic_space.space.Space.build(data="/tmp/temp_word2vec_model.dm", format="dm"))



    return (False, "Error :(")