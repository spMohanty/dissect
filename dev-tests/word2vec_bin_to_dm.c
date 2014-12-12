#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc/malloc.h> 
#include <stdbool.h>


//
// This converts a word2vec binary file to a .dm format which is used by the DISSECT toolkit for compositional semantic experiments.
//

const long long max_size = 2000;         // max length of strings
const long long N = 40;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f;
  char file_name[max_size];
  float len;
  long long words, size, a, b;
  char ch;
  float *M;
  char *vocab;
  if (argc < 2) {
    printf("Usage: ./a.out PATH_TO_WORD2VEC_TRAINED_BINARY_FILE_OF_THE_MODEL");
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &vocab[b * max_w], &ch);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) M[a + b * size] /= len;
  }
  fclose(f);
  // printf("%lld %lld #File: %s\n",words,size,file_name);

  // A Linear Transformation to make all the dimensions of all the feature vectors is necessary 
  // As DISSECT used NonNegative Matrix Factorization and hence doesnot support non negative values 
  //
  // And, a linear transformation is the best bet, as we do not want to loose any of the statistical properties of the distribution
  // by passing them through a funny function.

  // Do we will do two iterations in this pre-processing stage just to leave the statistical distribution of the params unaffected

  float *least_dim;
  least_dim = (float *)malloc(sizeof(float)*(long long)size);
  bool *is_initialized;
  is_initialized = (bool *)malloc(sizeof(bool)*(long long)size);
  for(b=0;b<size;b++){//Initialize the initialization flag vector to false
  	is_initialized[b] = false;
  }

  //First pass to estimate the bounds of the data points
  for(a=0; a<words; a++){
  	for(b=0;b<size; b++){
  		if(is_initialized[b] == false){
  			//Initialize the starting min value
  			least_dim[b] = M[a*size + b];
  			is_initialized[b] = true;
  		}else{
  			if(M[a*size + b] < least_dim[b]){
  				least_dim[b] = M[a*size + b];
  			}
  		}
  	}
  }

  //Seconds pass to print the translated vector space so that all values in all dimensions are non negative
  for (a = 0; a < words; a++){
    printf("%s ",&vocab[a * max_w]);
    for (b = 0; b< size; b++){ printf("%f ",(M[a*size + b])-least_dim[b]); }
    printf("\n");
  }  

  return 0;
}