from composes.similarity.cos import CosSimilarity
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.dim_reduction.svd import Svd
from composes.transformation.feature_selection.top_feature_selection import TopFeatureSelection
from composes.composition.lexical_function import LexicalFunction
from composes.composition.full_additive import FullAdditive
from composes.composition.weighted_additive import WeightedAdditive
from composes.composition.multiplicative import Multiplicative
from composes.composition.dilation import Dilation
from composes.utils.regression_learner import RidgeRegressionLearner

import composes.utils.io_utils as io_utils
import composes.utils.scoring_utils as scoring_utils
import composes.utils.space_utils as space_utils


#load a core space
print "Loading the data..."
data_path = "/Users/spmohanty/work/compositional_semantics/dissect/test_data/"

dissect_toy_data = "/Users/spmohanty/work/compositional_semantics/dissect/src/examples/data/"


## Only adding support for .dm files, as the sm files are inherently inefficiently designed 
## especially for latge vector spaces. :( 

space_file = "/Users/spmohanty/work/compositional_semantics/dissect/dev-tests/" + "word2vec_model.bin"
# space = Space.build(data = space_file,
#                        format = "dm")

(result, space) = space_utils.load_word2vec_bnary_as_space(space_file)
print result
print space

## Get similarity between words
print space.get_sim("cat","dog", CosSimilarity())
print space.get_sim("cat","car", CosSimilarity())

# # Get top-N neighbours

print space.get_neighbours("cat", 3, CosSimilarity())

## The concept of peripheral space is really interesting. Sadly we do not have any interesting ways 
# to create peripheral space to efficiently create a peripheral space from a random corpus.
# Just exposing the possibility to learn peripheral space if we have a corpus where phrases etc are marked 
# as one token. Then we can use the word2vec_bin_to_DISSECT_dm convertor to generate a similar dm
# print "Creating peripheral space.."
# per_space = PeripheralSpace.build(space,
#                                   format = "dm",
#                                   data ="SOME_PATH_FOR_A_WORD_TO_VEC_PERIPHERAL_SPACE_DATA"
#                                   )
		

# Debug
# print space.cooccurrence_matrix
# print space.id2row


# instantiate a weighted additive model
my_comp = WeightedAdditive(alpha = 1, beta = 1)

# use the model to compose words in my_space
composed_space = my_comp.compose([("good", "book", "good_book"),
                                  ("good", "car", "good_car")], 
                                 space)

print composed_space.id2row
print composed_space.cooccurrence_matrix
print composed_space.get_sims([("good_car","good_book")], CosSimilarity()) # Similarity metric

#===============================================================================================================
print "="*80
#===============================================================================================================
##Training Models

##Training Weight Additive Models

#training data
train_data = [("good", "car", "good_car"),
              ("good", "book", "good_book")
              ]

## Demonstrating on a smaller vector space for performance and demo
##	But the word2vec space can also be used, as it is just another object of the same class
#load an argument space
arg_space = io_utils.load(dissect_toy_data+"out/ex10.pkl")
print arg_space.id2row
print arg_space.cooccurrence_matrix

#load a phrase space
phrase_space = io_utils.load(dissect_toy_data+"out/PHRASE_SS.ex10.pkl")
print phrase_space.id2row
print phrase_space.cooccurrence_matrix

#train a weighted additive model on the data
my_comp = WeightedAdditive()
my_comp.train(train_data, arg_space, phrase_space)

#print its parameters
print "Weight Additive Model :: "
print "alpha:", my_comp.alpha
print "beta:", my_comp.beta

#===============================================================================================================
print "="*80
print "Time Dilation model"
#===============================================================================================================

#training data
train_data = [("good", "car", "good_car"),
              ("good", "book", "good_book")
              ]

#load an argument space
arg_space = io_utils.load(dissect_toy_data+"out/ex10.pkl")

#load a phrase space
phrase_space = io_utils.load(dissect_toy_data+"out/PHRASE_SS.ex10.pkl")
print "Training phrase space"
print phrase_space.id2row
print phrase_space.cooccurrence_matrix

#train a Dilation model on the data
my_comp = Dilation()
my_comp.train(train_data, arg_space, phrase_space)

#print its parameters
print "\nlambda:", my_comp._lambda

#use the model to compose the train data
composed_space  = my_comp.compose([("good", "bike", "good_bike")], 
                                  arg_space)
print "\nComposed space:" 
print composed_space.id2row
print composed_space.cooccurrence_matrix
#===============================================================================================================
print "="*80
print "Full Additive model"
#===============================================================================================================

#training data
train_data = [("good", "car", "good_car"),
              ("good", "book", "good_book")
              ]

#load an argument space
arg_space = io_utils.load(dissect_toy_data+"out/ex10.pkl")

#load a phrase space
phrase_space = io_utils.load(dissect_toy_data+"out/PHRASE_SS.ex10.pkl")
print "Training phrase space"
print phrase_space.id2row
print phrase_space.cooccurrence_matrix

#train a FullAdditive model on the data
my_comp = FullAdditive()
my_comp.train(train_data, arg_space, phrase_space)

#print its parameters
print "\nA:", my_comp._mat_a_t.transpose()
print "B:", my_comp._mat_b_t.transpose()

#use the model to compose the train data
composed_space  = my_comp.compose([("good", "bike", "good_bike")], 
                                  arg_space)
print "\nComposed space:" 
print composed_space.id2row
print composed_space.cooccurrence_matrix






