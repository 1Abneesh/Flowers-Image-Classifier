# Image Classifier Project
Developing an Image Classifier with Deep Learning

## Table of Contents
1. Overview
2. Part 1 - Jupyter Notebook Implementation
3. Part 2 - Command Line Application
4. Train
5. Predict
6. Specifications
7. Screenshot
8. Usage
9. Options
10. Deployment
11. Requirements
12. Contributing
13. License

## Overview
In this project, you'll train an image classifier to recognize different species of flowers. With the help of command line parameters you can enter variou parious parameter for yor model and make predictions.

## Part 1 - Jupyter Notebook Implementation
The project is broken down into multiple steps:
- Load and preprocess the image dataset
- Train the image classifier on your dataset
- Use the trained classifier to predict image content

## Part 2 - Command Line Application
Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a pair of Python scripts that run from the command line. For testing, you should use the checkpoint you saved in the first part.

## Train
The first file, `train.py`, will train a new network on a dataset and save the model as a checkpoint.

## Predict
The second file, `predict.py`, uses a trained network to predict the class for an input image.

## Specifications
The project submission must include at least two files `train.py` and `predict.py`. Make sure to include all files necessary to run `train.py` and `predict.py` in your submission.

## Screenshots
<p align="center">
  <img src="Training network.png">
</p>

<p align="center">
  <img src="Sanity Checking.png">
</p>

## Usage
To use the application, run the following commands:

# Train the network
python train.py data_directory

# Predict the class of an image
`python predict.py /path/to/image checkpoint`

# Options
The following options are available for both `train.py` and `predict.py`:

Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory` <br>
Choose architecture: `python train.py data_dir --arch "vgg13"`<br>
Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20` <br>
Use GPU for training: `python train.py data_dir --gpu` <br>
Return top K most likely classes: `python predict.py input checkpoint --top_k 3` <br>
Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json` <br>
Use GPU for inference: `python predict.py input checkpoint --gpu` <br>

## Deployment
The application can be deployed on any platform that supports Python.

## Requirements
The project requires the following packages: <br>

PyTorch <br>
Numpy <br>
Matplotlib <br>
Argparse <br>

## Contributing
If you wish to contribute to this project, please follow the standard Git workflow.

## License
The project is licensed under the `MIT` license.

