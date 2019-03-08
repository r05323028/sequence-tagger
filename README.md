# Sequence Tagger for Named Entity Recognition (TensorFlow)

This tagger is based on `tensorflow` and `fastText`. 
You need to prepare training, testing data and char, word embedding by yourself.
Now, the model is only support for binary classification i.e., `TARGET` and `OTHER` but you can easily modify the code to extend the model to multiclass version.

## Install requirements

```
pip install -r requirements.txt
```

## Data format

The data format we used in our model is as follows.
The dataset is a list of sentences which contain (segmented word, label) pairs.

```python
TRAINING_DATA = [
   [('吳卓源', 1), ('的', 0), ('星海', 0), ('完整', 0), ('的', 0), ('陳述', 0), ...],
   [('五月天', 1), ('即將', 0), ('在', 0), ('小巨蛋', 0), ('舉辦', 0), ...],
   ...
]
```

## Train

```
python train.py
```

## Tensorboard monitor

```
tensorboard --logdir=logs --host=<host> --port=<port>
```

## Reference

- <a href="https://github.com/guillaumegenthial/sequence_tagging">Named Entity Recognition (LSTM + CRF) - Tensorflow</a>

- <a href="https://github.com/shiyybua/NER">基于tensorflow深度学习的中文的命名实体识别</a>
