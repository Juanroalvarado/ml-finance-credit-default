import pandas as pd
import pickle

from xgb_model_functions import SplitModel
from preproc_functions import pre_process

from calibrator import calibrator

import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some named arguments.")
    
    # Define the named arguments
    parser.add_argument('--input_csv', type=str, required=True,)
    parser.add_argument('--output_csv', type=str, required=True)
    
    args = parser.parse_args()
    
    input_path = args.input_csv
    output_path = args.output_csv
    print(input_path)
    print(output_path)

    # Load pre computed data from training which is used for growth features
    historical_data = pd.read_csv('historical_features.csv')
    historical_data['stmt_date'] = pd.to_datetime(historical_data['stmt_date'])
    
    # custom bins for certain featured
    with open('custom_bins.pkl', 'rb') as inp:
        custom_bins = pickle.load(inp)
    # loads preprocessing parameters
    with open('preproc_params.pkl', 'rb') as inp:
        preproc_params = pickle.load(inp)
    # loads trained model
    with open('trained_model.pkl', 'rb') as inp:
        trained_model = pickle.load(inp)
    print('Loaded trained model')
    
    # read holdout data
    holdout_df = pd.read_csv(input_path,index_col=0)
    # set the right datetime format
    holdout_df['stmt_date'] = pd.to_datetime(holdout_df['stmt_date'])
    holdout_df['def_date'] = pd.to_datetime(holdout_df['def_date'], format="%d/%m/%Y")
    holdout_df.sort_values('stmt_date', inplace=True)
    # runs preprocessing functions
    test_data_proc , preproc_params = pre_process(holdout_df, 
                                             historical_df=historical_data, 
                                             new=False, 
                                             preproc_params = preproc_params,  
                                             quantiles = 50, 
                                             days_until_statement = 150)
    # calls out custom model predict function (XGB behind)
    predictions = trained_model.predict(test_data_proc)
    # calibrates predictions
    calibrated_predictions = calibrator(predictions)
    print("Predictions done")
    calibrated_predictions.to_csv(output_path,index=False,header=False)