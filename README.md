# CNN for Chineses characters classification

practice on CNN using pytorch

## Requirements

This is my experiment eviroument
- python 3.11
- pytorch 2.3.1
- torchvison 0.18.0
- tensorboard 2.17.0


## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar100
```

### 2. dataset
Dowload dataset from: https://drive.google.com/file/d/1AIi8286VY4QGJ8i2SY1sIQc7OeUdJqwX/view

Extract dataset and makesure that 952_test, 952_train, 952_val are in data folder

### 3. train the model
You need to specify the net you want to train using arg -net

Task1 and Task2:
```bash
# use gpu to train squeezenet
$ python train.py -net squeezenet -gpu
```
Task3:
```bash
# use gpu to train squeezenet
$ python c3_train.py -gpu
```

The supported net args are:
```
squeezenet
squeezenetlight
squeezenetv1
squeezenetv2
mobilenetv2
shufflenetv2
```


### 4. test the model
Test the model using test.py

Checkpoint are in checkpoint folder

Task1 and Task2:
```bash
$ python test.py -net squeezenet -weights path_to_squeezenet_weights_file
```

Task3:
```bash
$ python c3_test.py -weights path_to_weights_file
```

## Results
Task1 (99.76%) with squeezenetlight

Task2 (99.80%) with squeezenet

Task3 (98.69%) with squeezenetlight + LSTM

## Sorry for the bad implementation in Task 3 :((((