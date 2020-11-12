# QusetNet
## Introduction
This is a Tensorflow implmentation of "Is Deep Learning Effective for Quesionnaire Data Analysis?". This paper regards the questionnaire resonse information of each participant as a sequence. Use two kinds of experiments (i.e., demographic attribute classification and fake data detection) to explore whether deep learning is suitable for questionnaire analysis?

## Models
The current submission in the warehouse is the messy script we used to write the paper, code comments are not complete. We are currently reorganizing the code structure and will provide routines for demonstration later :)

## ToDo List
* [x] Release code
* [ ] Document for Installation
* [x] Trained models
* [x] Evaluation
* [ ] Demo script
* [ ] re-organize and clean the parameters
* [ ] tensorflow 2.0

## File structure
* `Demographic attribute classification`: Category related experiment code. 
* `Fake data detection`: Fake data detection experiment code. 

## Installation
You can install a suitable environment by running the following commands:
```bash
conda env create -f environment.yaml
```
### Requirements:
* Python 3.6.0+
* Tensorflow 1.12.0

## Thanks to the Third Party Libs
[Tensorflow](https://github.com/tensorflow/tensorflow/tree/r1.12)

[improved_wgan_training](https://github.com/igul222/improved_wgan_training)
