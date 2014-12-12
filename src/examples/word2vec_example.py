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

#load a core space
print "Loading the data..."
data_path = "/Users/spmohanty/work/compositional_semantics/dissect/test_data/"


## Only adding support for .dm files, as the sm files are inherently inefficiently designed 
## especially for latge vector spaces. :( 

space_file = data_path + "word2vec_trained_text8.dm"
space = Space.build(data = space_file,
                       format = "dm")

## Get similarity between words
print space.get_sim("cat","dog", CosSimilarity())
print space.get_sim("cat","car", CosSimilarity())

# Get top-N neighbours

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






