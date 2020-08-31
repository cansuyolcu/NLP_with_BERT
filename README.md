# NLP_with_BERT
Movies reviews Semantic analysis using BERT with ktrain library.

## Importing the libraries

```python
import os.path
import numpy as np
import tensorflow as tf
import ktrain
from ktrain import text
```
##  Data Preprocessing

**Loading the IMDB dataset**

```python
dataset = tf.keras.utils.get_file(fname="aclImdb_v1.tar.gz",
                                  origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                                  extract=True)
IMDB_DATADIR = os.path.join(os.path.dirname(dataset), 'aclImdb')
```

<img src= "https://user-images.githubusercontent.com/66487971/91720203-641a6400-eb9f-11ea-8207-0f5315a49d68.png" width = 800>




















































