What is in this directory:

--- training_and_validation:
- Estimation code is the "estimator_training.ipynb" this notebook just needs to have the input path to training data set up and then after running will processed the data, train and save the model and produce all necesarry components needed to run inference using the harness code.
- walk_forward_analysis.ipynb: this notebook will run the Walk Foward Analysis for different models and generates the visualizations for the modeling and evaluation sections in our presentation.



--- eda: this folder contains additional files which show our EDA process and generates the graphs found in our presentation


Running the harness:
- Contained in the root directory are the pickle objects that we have trained on the given train data, these include (custom_bins which we hand picked usual visual inspection, preproc_params which contain the PD values and bin cuts from the training that -- these are then used in the inference section, we do not recompute these values, and trained_model which is an XGBClassifier object)

- The harness.py when run in the current directory (after running estimaor) will process the input data, produce probabilities from the trainied model and write results -- usage: python3 harness.py --input_csv 'input.csv' --output_csv 'results.csv' 


----- Changes -------
1. We split the preprocessing function into one that is used for training (found in training_and_calidation directory) and the one in root called preproc_for_harness.py; here we removed all code that computes our PDs and features from the dataset, this harness specific preprocessing file takes the values from the preproc_params dictionary and maps them to the holdout data.
2. We removed our "growth" features. These where computed by combining the holdout data with computed features of the training data to check if we are seeing a firm again and could use previously seen data to compute this. We have removed this function despite us strongly believeing that there is no leakage there it could be misinsterpreted.
3. Removed some unused features from the preprocessing file.
4. Removed the function which computes our "default" label for training.
5. Trained a new XGBClassifier model to be used in all cases. Previously we used 2 models one for unseen and one for seen firms.
6. Some changes to how the calibrator outputs are constructed.



Github: https://github.com/Juanroalvarado/ml-finance-credit-default