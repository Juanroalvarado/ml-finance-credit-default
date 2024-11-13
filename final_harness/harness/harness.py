import pandas as pd
import pickle

from model_functions import SplitModel
from preproc_functions import pre_process
from statsmodels.iolib.smpickle import load_pickle

import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some named arguments.")
    
    # Define the named arguments
    parser.add_argument('--input', type=str, required=True,)
    parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    print(input_path)
    print(output_path)

    # Load pre computed data from training which is used for growth features
    historical_data = pd.read_csv('historical_data/historical_features.csv',index_col=0)
    historical_data['stmt_date'] = pd.to_datetime(historical_data['stmt_date'])
    
    # custom bins for certain featured
    with open('model_objs/custom_bins.pkl', 'rb') as inp:
        custom_bins = pickle.load(inp)
    # bin PDs from historical
    with open('model_objs/preproc_params.pkl', 'rb') as inp:
        preproc_params = pickle.load(inp)
    # # trained model
    print('Loading trained model')
    # with open('first_model.pkl', 'rb') as inp:
    #     trained_model = pickle.load(inp)
    trained_model = load_pickle('model_objs/trained_model.pkl')

    # read and process holdout
    holdout_df = pd.read_csv(input_path,index_col=0)

    holdout_df['stmt_date'] = pd.to_datetime(holdout_df['stmt_date'])
    holdout_df['def_date'] = pd.to_datetime(holdout_df['def_date'], format="%d/%m/%Y")
    holdout_df.sort_values('stmt_date', inplace=True)

    test_data_proc , preproc_params = pre_process(holdout_df, 
                                             historical_df=historical_data, 
                                             new=False, 
                                             preproc_params = preproc_params,  
                                             quantiles = 50, 
                                             days_until_statement = 150)

    predictions = trained_model.predict(test_data_proc)
    print("Predictions done")
    predictions.to_csv(output_path,index=False,header=False)