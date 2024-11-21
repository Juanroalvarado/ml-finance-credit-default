import pandas as pd
import pickle

from xgboost import XGBClassifier
from preproc_for_harness import pre_process

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
    
    # custom bins for certain featured
    with open('custom_bins_v2.pkl', 'rb') as inp:
        custom_bins = pickle.load(inp)
    # loads preprocessing parameters
    with open('preproc_params_v2.pkl', 'rb') as inp:
        preproc_params = pickle.load(inp)
    # loads trained model
    with open('trained_model_v2.pkl', 'rb') as inp:
        trained_model = pickle.load(inp)
    print('Loaded trained model')
    
    # read holdout data
    holdout_df = pd.read_csv(input_path,index_col=0)
    # set the right datetime format
    holdout_df['stmt_date'] = pd.to_datetime(holdout_df['stmt_date'])
    # holdout_df['def_date'] = pd.to_datetime(holdout_df['def_date'], )
    # holdout_df.sort_values('stmt_date', inplace=True) # can be removed
    # runs preprocessing functions
    test_data_proc , preproc_params = pre_process(holdout_df, 
                                             # historical_df=historical_data, 
                                             preproc_params = preproc_params)
    ### features for first firm
    features = [
        'financial_leverage_quantile_values',
        'profitability_ratio_quantile_values',
        'quick_ratio_v2_quantile_values',
        'dscr_quantile_values',
        'roe_quantile_values',
        'cfo_quantile_values',
        'regional_code_pd' 
    ]
    # calls out custom model predict function (XGB behind)
    predictions = trained_model.predict_proba(test_data_proc[features])[:,1]
    # calibrates predictions
    calibrated_predictions = pd.Series(calibrator(predictions))
    print("Predictions done")
    calibrated_predictions.to_csv(output_path,index=False,header=False)