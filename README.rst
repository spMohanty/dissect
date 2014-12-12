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