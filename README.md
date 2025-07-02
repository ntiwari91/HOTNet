# HOTNet
The code is for the work:

```

```



## Requirements

``` python
pytorch == 1.6.1

```

### Dataset

To train and test on CAVE data set, you must first download the CAVE data set form http://www.cs.columbia.edu/CAVE/databases/multispectral/. Put all the training images and test images in their respective folders. You can also download the processed data from https://drive.google.com/drive/folders/1lwsNkmDFW81PvRGPWWBh-5wQDtF8XgQ5?usp=sharing 

## Train

```python
python main.py --mode train
```



## Test

```python
python main.py --mode test --nEpochs 150
```



