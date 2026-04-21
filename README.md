# MNIST Neural Network Trainer

A MLP neural network to train on and recognize digits using the MNIST dataset.

## Overview

- **Neural Network**
- **Asynchronous batch training** to prevent freezing during Editor play
- **MNIST binary parser** for loading images and labels

## Features

### Core Architecture
- **Forward Pass & Backpropagation**
- **Configuration** : Supports configurable hidden layer sizes

### Training UI Integration
- **Batch Processing** : Processes user-defined chunks per tick
- **Real-time Progress** : Returns precise percentage in widget bar
- **Accuracy Evaluation** : Runs predictions on unseen data to calculate model accuracy

## Configuration Parameters

### Initialization Settings (`InitTrainer`)

**Network Topology :**
- `HiddenNodes` : Number of neurons in the hidden layer
*(Input is fixed at 784 pixels, Output is fixed at 10 digits)*

- `InLearningRate` : Step size for weight adjustments
- `InEpochs` : Number of times the network loop over the entire dataset

### Tick Execution Settings (`TrainBatch`)
- `BatchSize` : Number of images to process per frame

### Validation Settings
- `NumTestsReserved` : Number of images from the 60k pool for accuracy testing.

## Usage

**On a user widget launch at level starting :**
- Button `Train` to toggle train
- Percentage bar for show progression
- Result of the accuracy prediction (/1)

*On other hand, you can test some other methods like XOR learning etc... in the level blueprint (results on unreal engine logs)*