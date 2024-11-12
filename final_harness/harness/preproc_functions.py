import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import warnings

def calculate_default_within_year(df, stmt_date_col='stmt_date', def_date_col='def_date', days_until_statement=150):
    """
    Calculate a default status within an adjusted timeframe and create a column indicating this in the DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing statement and default dates.
    - stmt_date_col (str): Column name for the statement date.
    - def_date_col (str): Column name for the default date.
    - days_until_statement (int): Number of days to add to the statement date before calculating the one-year window. Default is 150.
    
    Returns:
    - float: The mean of the 'default_within_timeframe' column, representing the proportion of defaults within the adjusted one-year period.
    """
    # Convert columns to datetime
    df[stmt_date_col] = pd.to_datetime(df[stmt_date_col])
    df[def_date_col] = pd.to_datetime(df[def_date_col])
    
    # Adjust the statement date by adding days_until_statement
    adjusted_stmt_date = df[stmt_date_col] + pd.DateOffset(days=days_until_statement)
    
    # Calculate one year after the adjusted statement date
    one_year_after = adjusted_stmt_date + pd.DateOffset(years=1)
    
    # Define default status within the adjusted one-year period and create the column
    df['default'] = ((df[def_date_col] >= adjusted_stmt_date) & (df[def_date_col] <= one_year_after)).astype(int)
    
    # Return the mean of the default_within_timeframe column
    return df

def make_quantiles(df, field, num_quantiles=4, custom_bins = {}, new_column_name=None, new=True, preproc_params = None):
    """
    Creates a new column in the DataFrame indicating the quantile for each row based on a specified field.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - field (str): The column name of the field to compute quantiles for.
    - num_quantiles (int): The number of quantiles to divide the field into (default is 4 for quartiles).
    - new_column_name (str): The name for the new column to store the quantile information.
                             If None, defaults to '{field}_quantile'.
    
    Returns:
    - pd.DataFrame: The DataFrame with the new quantile column added.
    """
     # Set the new column name if not provided
    if new_column_name is None:
        new_column_name = f"{field}_quantile"
        
    if new:
        if new_column_name in custom_bins:
            print(f'Custom bins for {new_column_name}')
            bins = custom_bins[new_column_name]
            bins = [-np.inf]+bins+[np.inf]
            
            data_to_cut = df[field].replace([np.inf, -np.inf], [df[field][np.isfinite(df[field])].max(), df[field][np.isfinite(df[field])].min()])
            cut = pd.cut(data_to_cut, bins=bins, labels=False, duplicates = 'drop', include_lowest=True) + 1  # Adding 1 to make quantiles start from 1
        else:
            
            data_to_cut = df[field].replace([np.inf, -np.inf], [df[field][np.isfinite(df[field])].max(), df[field][np.isfinite(df[field])].min()])
            cut, bins = pd.qcut(data_to_cut, q=num_quantiles, labels=False, retbins=True, duplicates = 'drop')
            bins = [-np.inf]+bins.tolist()+[np.inf]

            cut = pd.cut(data_to_cut, bins=bins, labels=False, duplicates = 'drop', include_lowest=True) + 1  # Adding 1 to make quantiles start from 1
    
        
        # Calculate the quantiles and create the new column
        df[new_column_name] = cut
        prob_values = df.groupby(cut)[['default']].mean()
        
        df[f'{new_column_name}_values'] = cut.to_frame().merge(prob_values, on=field, how='left')['default'].fillna(0.01).values

        preproc_params['quantile_bins'][new_column_name] = bins
        preproc_params['quantile_values'][new_column_name] = prob_values
        
    else:
        print(new_column_name)
        bins = preproc_params['quantile_bins'][new_column_name]
        prob_values = preproc_params['quantile_values'][new_column_name]

        cut = pd.cut(df[field], bins=bins, labels=False, duplicates = 'drop', include_lowest=True) + 1  # Adding 1 to make quantiles start from 1
        # print(cut)
        # print(cut.to_frame())
        # print(prob_values)
        df[new_column_name] = cut
        df[f'{new_column_name}_values'] = cut.to_frame().merge(prob_values, on=field, how='left')['default'].fillna(0.01).values
        
    return df, preproc_params

def create_growth_features(df, id_col, date_col, field,  historical_df = None, new=True, ):
    """
    Creates a growth feature and its quantiles based on percentage change in the specified field, 
    grouped by ID and sorted by date.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - id_col (str): The column name for the unique identifier (e.g., 'id').
    - date_col (str): The column name for the date to sort by (e.g., 'stmt_date').
    - field (str): The column name for which to calculate the growth feature.

    Returns:
    - pd.DataFrame: DataFrame with the growth feature and its quantiles added.
    """
    if new:
        df = df.sort_values(by=[id_col, date_col])
    
        # Calculate percentage change for the growth feature
        growth_feature = f"{field}_growth"
        df[growth_feature] = df.groupby(id_col)[field].pct_change()
        
        # Fill missing values (first occurrence per group) with 0
        df[growth_feature] = df[growth_feature].fillna(0)

        df['is_first_occurrence'] = (df['id'] != df['id'].shift()).astype(int)
    else:
        # join historical and testing data frame
        # Sort by ID and date to calculate growth correctly
        concat_df = pd.concat([df, historical_df]).sort_values(by=[id_col, date_col])

        concat_df['is_first_occurrence'] = (concat_df['id'] != concat_df['id'].shift()).astype(int)
        
        # Calculate percentage change for the growth feature
        growth_feature_name = f"{field}_growth"
        growth_features = concat_df.groupby(id_col)[field].pct_change()

        df = df.join(growth_features.to_frame(growth_feature_name), how='left')
        if 'is_first_occurrence' not in df.columns:
            df = df.join(concat_df[['is_first_occurrence']], how='left')

        
        
        # Fill missing values (first occurrence per group) with 0
        # df[growth_feature].fillna(0, inplace=True)
    return df

def data_imputation(df):
    # Condition to check if EBITDA is missing or zero and prof_operations is not
    df['ebitda'] = np.where((df['ebitda'].isna() | (df['ebitda'] == 0)) & df['prof_operations'].notna() & (df['prof_operations'] != 0),
                            df['prof_operations'], df['ebitda'])

    # Condition to check if prof_operations is missing or zero and EBITDA is not
    df['prof_operations'] = np.where((df['prof_operations'].isna() | (df['prof_operations'] == 0)) & df['ebitda'].notna() & (df['ebitda'] != 0),
                                     df['ebitda'], df['prof_operations'])
    
    # if we have operating profit, and profit shows 0, set to operating profit instead
    df['profit'] = df.apply(lambda row: 1 if row['profit'] == 0 and row['prof_operations'] == 0
                            else row['prof_operations'] if row['profit'] == 0 and row['prof_operations'] != 0
                            else row['profit'], axis=1)
    #if all else fails, impute ebitda based on ratio of total assets
    df['ebitda'] = df['ebitda'].fillna(df['asst_tot'] * 0.05)
    df['cash_and_equiv'] = df['cash_and_equiv'].fillna(df['asst_tot'] * 0.05)
    df['profit'] = df['profit'].fillna(0.01)

    #do roe
    df['roe'] = df['profit'] / df['eqty_tot']
    df['roe'] = df['roe'].fillna(0)

    # Define if we have ebitda and operating rev is 0, set to ebitda
    df['rev_operating'] = df.apply(lambda row: 
                                 row['ebitda'] if row['rev_operating'] == 0 and row['ebitda'] > 0 
                                 else 1 if row['rev_operating'] == 0 and row['ebitda'] <= 0
                                 else row['rev_operating'], axis=1)
    df['rev_operating'] = df['rev_operating'].fillna(df['ebitda']).fillna(1)
    
    return df

def pre_process(df, historical_df=None, custom_bins = None, new=True, preproc_params = None, quantiles = 10, days_until_statement = 150):
    """
    Preprocesses 

    Parameters:
    - df (pd.DataFrame): The input DataFrame 

    Returns:
    - pd.DataFrame: The DataFrame with new features and quantiles added.
    """
    # Impute missing values
    df = data_imputation(df)

    # Calculate default status within one year
    df = calculate_default_within_year(df, days_until_statement=days_until_statement)
    
    # Create quantiles for total assets
    df, preproc_params = make_quantiles(df, field='asst_tot', num_quantiles=quantiles, new=new, preproc_params=preproc_params)

    # Calculate total liabilities and financial leverage ratio (assume debts are zero if left na)
    df['liab_tot'] = np.where(df['debt_st'].isna(), 0, df['debt_st'])  + np.where(df['debt_lt'].isna(), 0, df['debt_lt'])
    df['financial_leverage'] = df['liab_tot'] / df['asst_tot']
    df, preproc_params = make_quantiles(df, field='financial_leverage', custom_bins=custom_bins, num_quantiles=quantiles, new=new, preproc_params=preproc_params)

    # Calculate profitability ratio
    df['profitability_ratio'] = df['profit'] / df['asst_tot']
    df, preproc_params = make_quantiles(df, field='profitability_ratio', custom_bins=custom_bins, num_quantiles=quantiles, new=new, preproc_params=preproc_params)

    df, preproc_params = make_quantiles(df, field='roe', custom_bins=custom_bins, num_quantiles=quantiles, new=new, preproc_params=preproc_params)

    # Calculate net income growth by ID and sort by statement date
    df['net_income'] = df['profit']
    df = create_growth_features(df, historical_df = historical_df, new=new, id_col='id', date_col='stmt_date', field='net_income')
    df, preproc_params = make_quantiles(df, field='net_income_growth', custom_bins=custom_bins, num_quantiles=quantiles, new=new, preproc_params=preproc_params)

    # Calculate Quick Ratio Version 2
    df['quick_ratio_v2'] = np.where(df['debt_st'] == 0, 100, (df['cash_and_equiv'] + df['AR']) / df['debt_st'])
    #fill with median, mean is too high, skewed
    df['quick_ratio_v2'] = df['quick_ratio_v2'].fillna(df['asst_tot'] * 0.8)
    df, preproc_params = make_quantiles(df, field='quick_ratio_v2', num_quantiles=quantiles, new=new, preproc_params=preproc_params)

    # Calculate sales growth by ID and sort by statement date
    df['sales'] = df['rev_operating']
    df = create_growth_features(df, historical_df = historical_df, new=new, id_col='id', date_col='stmt_date', field='sales')
    df, preproc_params = make_quantiles(df, field='sales_growth', num_quantiles=quantiles, new=new, preproc_params=preproc_params)

    # Calculate cash-assets ratio
    df['cash_assets_ratio'] = df['cash_and_equiv'] / df['asst_tot']
    df, preproc_params = make_quantiles(df, field='cash_assets_ratio', num_quantiles=quantiles, new=new, preproc_params=preproc_params)

    # Calculate Debt Service Coverage Ratio (DSCR) 
    # (roughly mean financing expense but not 0 for imputation)
    df['exp_financing'] = df['exp_financing'].replace(0, 10000)
    df['exp_financing'] = df['exp_financing'].fillna(10000)
    df['dscr'] = df['ebitda'] / df['exp_financing']
    df, preproc_params = make_quantiles(df, field='dscr', num_quantiles=quantiles, new=new, preproc_params=preproc_params)
    

    return df, preproc_params