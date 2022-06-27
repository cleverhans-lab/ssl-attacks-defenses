# On the Difficulty of Defending Self-Supervised Learning against Model Extraction


## Abstract 

Self-Supervised Learning (SSL) is an increasingly popular ML paradigm that trains models to transform complex inputs into representations without relying on explicit labels.These representations encode similarity structures that enable efficient learning of multiple downstream tasks. Recently, ML-as-a-Service providers have commenced offering trained SSL models over inference APIs, which transform user inputs into useful representations for a fee. However, the high cost involved to train these models and their exposure over APIs both make black-box extraction a realistic security threat. We thus explore model stealing attacks against SSL. Unlike traditional model extraction on classifiers that output labels, the victim models here output representations; these representations are of significantly higher dimensionality compared to the low-dimensional prediction scores output by classifiers. We construct several novel attacks and find that approaches that train directly on a victim’s stolen representations are query efficient and enable high accuracy for downstream models. We then show that existing defenses against model extraction are inadequate and not easily retrofitted to the specificities of SSL.

## Training a Victim Model

To train a victim model using the SimCLR method use the ```run.py``` file with the appropriate parameters set. Example: 

```python

$ python run.py -data ./datasets --dataset 'cifar10' --epochs 200 --arch 'resnet34' 

```

The ```run.py``` file calls the file ```simclr.py``` which contains the specific function for training a victim model.

## Stealing from a Victim Model

To steal from a victim model,  ```steal.py``` file with the appropriate parameters set setting the number of queries, the loss function etc. Example: 

```python

$ python steal.py -data ./datasets --dataset 'cifar10' --datasetsteal 'svhn'  --epochs 100 --arch 'resnet34' --losstype 'mse' --num_queries 9000

```

The ```steal.py``` file calls the file ```simclr.py``` which contains the specific function for stealing from a victim model. The file ```loss.py``` contains the implementation of various loss functions which are called by ```simclr.py```

## Linear Evaluation

Feature evaluation is done using a standard linear evaluation protocol. The file ```linear_eval.py``` is used for this evaluation for both a victim model and a stolen model. Example for how to run this file: 

```python

$ python linear_eval.py --dataset 'cifar10' --datasetsteal 'svhn' --dataset-test 'stl10' --losstype 'mse' --num_queries 9000

```


## Imagenet Victim Model

To steal from an imagenet victim model and run the associated linear evaluation, the files ```stealsimsiam.py``` and ```linsimsiam.py``` are used respectively. The victim model can be downloaded from the SimSiam Github repository: https://github.com/facebookresearch/simsiam. Examples of running these files:

```python
python stealsimsiam.py --world-size -1 --rank 0 --pretrained models/checkpoint_0099-batch256.pth.tar --batch-size 512 --lr 0.1 --lars --losstype 'infonce' --datasetsteal 'imagenet' --workers 8 --num_queries 100000 --temperature 0.1
python linsimsiam.py --world-size -1 --rank 0 --batch-size 256 --lars --lr 1.0 --datasetsteal 'imagenet' --num_queries 250000 --losstype 'infonce' --dataset 'cifar10' --modeltype 'stolen' --workers 8
```


## Defenses

Several defense approaches are present within the files (watermarking, prediction poisoning).

Note: Parts of the code are based off the following Github repositories: https://github.com/sthalles/SimCLR, https://github.com/facebookresearch/simsiam

### Citing this work
If you use this repository for academic research, you are highly encouraged (though not required) to cite our paper:
```
@inproceedings{sslextractions2022icml,
  title = {On the Difficulty of Defending Self-Supervised Learning against Model Extraction},
  author = {Dziedzic, Adam and Dhawan, Nikita and Kaleem, Muhammad Ahmad and Guan, Jonas and Papernot, Nicolas},
  booktitle = {ICML (International Conference on Machine Learning)},
  year = {2022}
}
```