What is in this directory:

- Estimation code is the "estimator_training.ipynb" this notebook just needs to have the input path to training data set up and then after running will processed the data, train and save the model and produce all necesarry components needed to run inference using the harness code.
- The harness.py when run in the current directory (after running estimaor) will process the input data, produce probabilities from the trainied model and write results -- usage: python3 harness.py --input_csv 'input.csv' --output_csv 'results.csv' 

- lavender_harness: this folder is what we recommend be used for the horse race. It contains the saved objects used for estimation, it follows the structure of what we sent to test our harness and includes poetry file to install our custom enviroment.

- eda: this folder contains additional files which show our EDA process and generates the graphs found in our presentation

- walk_forward_analysis.ipynb: this notebook will run the Walk Foward Analysis for different models and generates the visualizations for the modeling and evaluation sections in our presentation.


Github: https://github.com/Juanroalvarado/ml-finance-credit-default