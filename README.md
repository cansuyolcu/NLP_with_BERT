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

<img src= "https://user-images.githubusercontent.com/66487971/91720203-641a6400-eb9f-11ea-8207-0f5315a49d68.png" width = 650>

```python
print(os.path.dirname(dataset))
print(IMDB_DATADIR)
```

<img src= "https://user-images.githubusercontent.com/66487971/91720351-a93e9600-eb9f-11ea-91b1-48174c24fe1c.png" width = 500>

**Creating the training and test sets**

```python
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(datadir=IMDB_DATADIR,
                                                                       classes=['pos','neg'],
                                                                       maxlen=500,
                                                                       train_test_names=['train','test'],
                                                                       preprocess_mode='bert')
```

<img src= "https://user-images.githubusercontent.com/66487971/91720482-e571f680-eb9f-11ea-890c-d1097bd4f7e1.png" width = 500>

**Building the BERT model**

```python

model = text.text_classifier(name='bert',
                             train_data=(x_train, y_train),
                             preproc=preproc)
                             
```

<img src= "https://user-images.githubusercontent.com/66487971/91720640-2538de00-eba0-11ea-84c6-a366614515f9.png" width = 500>




**Training the BERT model**

```python

learner = ktrain.get_learner(model=model,
                             train_data=(x_train, y_train),
                             val_data=(x_test, y_test),
                             batch_size=6)
```

```python
learner.fit_onecycle(lr=2e-5,
                     epochs=1)
```

<img src= "https://user-images.githubusercontent.com/66487971/91720849-7648d200-eba0-11ea-8250-1822251dc46f.png" width = 800>



















































