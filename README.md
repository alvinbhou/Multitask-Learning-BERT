# Multitask Learning with BERT

### [Demo Website: Natural Language Understanding](http://aics.nctu.me:3000)

This repo demos the concept of Multitask Learning with BERT. According to the original paper, there are 4 different downstream applications for BERT. We fine tune the `Sentence Classification` and `Sentence Tagging` two tasks jointly. 

<img src="https://i.imgur.com/2cNNZxA.png" width="400px">

This type of model structure could be used in tasks when word tags and sentence labels have high correlation. Since this repos is a proof-of-concept demo, you can view the actual implementation of [Speech Act](https://en.wikipedia.org/wiki/Speech_act) tagging + classification [here](http://aics.nctu.me:3000)




## Model Structure
### Baseline model
<img src="https://i.imgur.com/HPEkpXB.png" width="400px">

### Improved model adding CRF layer
<img src="https://i.imgur.com/6nqVJCZ.png" width="400px">

> BERT image reference: http://jalammar.github.io/illustrated-bert/


## Performance on Speech Act Classification and Tagging
| Model |  Classification Acc.  | Tagging Acc. | Tagging F1|Tagging Precision|Tagging Recall|
| -------- | -------- | -------- | -------- | -------- |-------- |
| BERT  | **94.38**       | **99.21**    |92.68     | **93.97** | 91.84|
| BERT+CRF  | 93.85       | **99.21**   |**92.82**     |93.89| 91.77|
| BERT-WWM+CRF  | 93.65      | 99.20  |92.77   |93.57| **92.00**|

#### Dataset Statistics
All: 105116 sentences 

Speech Act label size: 20

## Training

We do not include the training code in this repo. Just a simple demonstration how it works.

#### Example Usage
```
python main.py
    --logdir logs/log1
    --lr 0.0001 
    --n_epochs 40
    --seed 1234 
    --batch_size 256
    --train_path data/data_train.txt 
    --eval_path data/data_eval.txt  
```

#### All Arguments
```
usage: main.py [-h] [--batch_size BATCH_SIZE] [--lr LR] [--n_epochs N_EPOCHS]
               [--test] [--crf] [--logdir LOGDIR] [--model_path MODEL_PATH]
               [--train_path TRAIN_PATH] [--eval_path EVAL_PATH]
               [--n_worker N_WORKER] [--seed SEED] [--sent SENT]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size
  --lr LR               learning rate
  --n_epochs N_EPOCHS   number of epochs
  --test                start testing process
  --crf                 use BERT-CRF model
  --logdir LOGDIR       log directory for saving/monitor model
  --model_path MODEL_PATH
                        pretrained model path
  --train_path TRAIN_PATH
                        training data path
  --eval_path EVAL_PATH
                        evaluation data path
  --n_worker N_WORKER   number of workers
  --seed SEED           torch manual seed
  --wwm                 use whole word mask model
  --sent SENT           test sentence to be tagged/classified
```

## Inference
#### Predict with trained model

```bash

$ python main.py --test --model_path=/path/to/model.pt --sent "Today is a beautiful day."

$ python main.py --test --model_path=/path/to/model.pt --sent "你為什麼要去實習？"

"""
('predicted_sentence_class',
 ['word_tag1', 'word_tag2', 'word_tag3', 'word_tag4', 'word_tag5'])

('ques_why', ['O', 'ques_when', 'ques_when', 'ques_why', 'O', 'O', 'O', 'O', 'O'])
"""
```

#### RESTful API
```bash
$ python application.py  # start web server

$ curl --header "Content-Type: application/json" \
  --request POST \
  --data '{"sent":"你要回家嗎"}' \
  localhost:3000/sact

>>> ["ques_yn",["O","O","O","O","O"]]
```

## PyTest testing
```bash
$ pytest

collected 5 items                                         
tests/test_bert.py ..                         
tests/test_bert_crf.py ..
tests/test_metrics.py .
=== 5 passed, 7 warnings in 27.47 seconds ===
```
