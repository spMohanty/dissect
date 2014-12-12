DIStributional SEmantics Composition Toolkit 
============================================


For documentation, please, refer to http://clic.cimec.unitn.it/composes/toolkit/


Support for word2vec binaries
============================
Word2Vec binaries can be converted into a .dm format using ```./dev/tests/word2vec_bin_to_dm.c```   

```
cd dev-tests
gcc word2vec_bin_to_dm.c
./a.out PATH_TO_WORD_2_VEC_BINARY > word2vec.dm
```

or you can also directly use the python interface to import the binary model in your python code.   
But NOTE : The python implementation will be slower as it doesnot currently use Cython for an optimized way to process the binary file
```
import composes.utils.space_utils as space_utils

space_file = "PATH_TO/"+ "word2vec_model.bin"
# result can be True or False depending on if it was successful
# the binary format released by the google folks has changed the few times(depending on the internal version they use), so you never know :(
(result, space) = space_utils.load_word2vec_bnary_as_space(space_file)

#Now this is a Space class object of DISSECT, and hence you use all the available composition models, etc on it
#Refer to word2vec_example.py for more detailed usage instructions

```


Installation
============
```
git clone https://github.com/spMohanty/dissect
cd dissect
python setup.py install
```

Usage
=====
You can run the test example using 
```
cd dissect
python src/examples/word2vec_example.py
```