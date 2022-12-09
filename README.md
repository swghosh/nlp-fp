# CS388 Final Project: Analyzing and Mitigating the Effect of Different Pre-trained Transformer Models for NLP Question Answering

This code was adapted from https://github.com/gregdurrett/fp-dataset-artifacts. More details about getting started with this code could be found in the [original readme](./old_README.md).

Enlisted below are a few set of experimental runs on different transformer models (primarily pre-trained from a curated set of models obtained from [Hugging Face Transformers Hub](https://huggingface.co/models)) for training on Question Answering task.

## Changing Optimizers and Learning Rate Schedule
- Using ElectraSmall model with SGD optimizer and a learning rate with warmup and linear schedule.
```bash
$ python3 run.py --do_train --task qa --dataset squad --output_dir /tmp/model1 --optim sgd
---
$ python3 run.py --do_eval --task qa --dataset squad --model /tmp/model1 --output_dir /tmp/model_eval
---
{'eval_exact_match': 2.374645222, 'eval_f1': 2.374645222}
```

- Using ElectraSmall model with SGD optimizer cosine restarts learning rate schedule a.k.a SGDR.
```bash
$ python3 run.py --do_train --task qa --dataset squad --output_dir /tmp/model1 --optim sgd --lr_scheduler_type cosine_with_restarts
---
$ python3 run.py --do_eval --task qa --dataset squad --model /tmp/model1 --output_dir /tmp/model_eval
---
{'eval_exact_match': 1.967833491, 'eval_f1': 9.816364186}
```


- Using ElectraSmall model with Adafactor optimizer and a learning rate with warmup and linear schedule.
```bash
$ python3 run.py --do_train --task qa --dataset squad --output_dir /tmp/model1 --optim adafactor
---
$ python3 run.py --do_eval --task qa --dataset squad --model /tmp/model1 --output_dir /tmp/model_eval
---
{'eval_exact_match': 78.4011352885525, 'eval_f1': 86.19613744}
```

## Changing Models

- Using ElectraSmall model with AdamW optimizer and a learning rate with warmup and linear schedule.
```bash
$ python3 run.py --do_train --task qa --dataset squad --output_dir /tmp/model --model google/electra-small-discriminator
---
$ python3 run.py --do_eval --task qa --dataset squad --model /tmp/model --output_dir /tmp/eval
---
Evaluation results:
{'eval_exact_match': 78.6092715231788, 'eval_f1': 86.29305652307012}
```

- Using BERT model with AdamW optimizer and a learning rate with warmup and linear schedule.
```bash
$ python3 run.py --do_train --task qa --dataset squad --output_dir /tmp/model --model bert-base-uncased
---
$ python3 run.py --do_eval --task qa --dataset squad --model /tmp/model --output_dir /tmp/eval
---
Evaluation results:
{'eval_exact_match': 81.1826,	'eval_f1': 88.4584}
```

- Using  DistilRoBERTa model with AdamW optimizer and a learning rate with warmup and linear schedule.
```bash
$ python3 run.py --do_train --task qa --dataset squad --output_dir /tmp/model --model distilroberta-base
---
$ python3 run.py --do_eval --task qa --dataset squad --model /tmp/model --output_dir /tmp/eval
---
Evaluation results:
{'eval_exact_match': 81.66508988,	'eval_f1': 88.74311131}
```

## Project Contributors
- [Abhilasha Singh](https://github.com/AbhilashaSingh)
- [Swarup Ghosh](https://github.com/swghosh)
