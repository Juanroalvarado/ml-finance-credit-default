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
            bins = [-np.inf]+bins[1:-1]+[np.inf]
           
            data_to_cut = df[field]#.replace([np.inf, -np.inf], [df[field][np.isfinite(df[field])].max(), df[field][np.isfinite(df[field])].min()])
            cut = pd.cut(data_to_cut, bins=bins, labels=False, duplicates = 'drop', include_lowest=True) + 1  # Adding 1 to make quantiles start from 1
        else:
           
            data_to_cut = df[field]#.replace([np.inf, -np.inf], [df[field][np.isfinite(df[field])].max(), df[field][np.isfinite(df[field])].min()])
            cut, bins = pd.qcut(data_to_cut, q=num_quantiles, labels=False, retbins=True, duplicates = 'drop')
            bins = [-np.inf]+bins.tolist()[1:-1]+[np.inf]

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

def calculate_conditional_pd_for_categorical(df, field, default_col='default', new=True, preproc_params=None):
    """
    Calculate the conditional probability of default for each unique category in a categorical field.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - field (str): The categorical column name for which to calculate conditional PD.
    - default_col (str): The column name for the default indicator (1 for default, 0 for non-default).
    
    Returns:
    - pd.DataFrame: The DataFrame with a new column for conditional PD values by category.
    """
    if new:
        # Create a new column name for storing default probabilities
        prob_column_name = f"{field}_pd"
        
        # Calculate the default probability for each unique category
        category_default_prob = df.groupby(field)[default_col].mean()
        
        # Map the default probability to each row based on its category
        df[prob_column_name] = df[field].map(category_default_prob)
        
        # Optionally store the probabilities in preproc_params if needed for later
        
        preproc_params['category_pd'][f'{field}_pd_values'] = category_default_prob
    else:
        print(f'using training pds for {field}')
        category_default_prob = preproc_params['category_pd'][f'{field}_pd_values']
        df[f'{field}_pd'] = df[field].map(category_default_prob).fillna(0.01)
    
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

        df['is_first_occurrence'] = (df[id_col] != df[id_col].shift()).astype(int).values
    else:
        # join historical and testing data frame
        # Sort by ID and date to calculate growth correctly
        concat_df = pd.concat([df, historical_df]).sort_values(by=[id_col, date_col])

        concat_df['is_first_occurrence'] = (concat_df[id_col] != concat_df[id_col].shift()).astype(int).values
       
        # Calculate percentage change for the growth feature
        growth_feature_name = f"{field}_growth"
        concat_df[growth_feature_name] = concat_df.groupby(id_col)[field].pct_change().values

        df = df.merge(concat_df[[growth_feature_name, id_col,date_col]], on=[id_col,date_col], how='left')
        
        if 'is_first_occurrence' not in df.columns:
            # df = df.join(concat_df[['is_first_occurrence']], how='left')
            df = df.merge(concat_df[['is_first_occurrence', id_col,date_col]], on=[id_col,date_col], how='left')

       
       
        # Fill missing values (first occurrence per group) with 0
        df[growth_feature_name] = df[growth_feature_name].fillna(0)
    return df

def data_imputation(df):
    # Condition to check if EBITDA is missing or zero and prof_operations is not
    df['ebitda'] = np.where((df['ebitda'].isna() | (df['ebitda'] == 0)) & df['prof_operations'].notna() & (df['prof_operations'] != 0),
                            df['prof_operations'], df['ebitda'])

    # Condition to check if prof_operations is missing or zero and EBITDA is not
    df['prof_operations'] = np.where((df['prof_operations'].isna() | (df['prof_operations'] == 0)) & df['ebitda'].notna() & (df['ebitda'] != 0),
                                     df['ebitda'], df['prof_operations'])
    
    # if we have operating profit, and profit shows 0, set to operating profit instead
    # df['profit'] = df.apply(lambda row: 1 if row['profit'] == 0 and row['prof_operations'] == 0
    #                         else row['prof_operations'] if row['profit'] == 0 and row['prof_operations'] != 0
    #                         else row['profit'], axis=1)
    df['profit'] = np.where((df['profit'] == 0) & (df['prof_operations'] == 0), 100,
                    np.where((df['profit'] == 0) & (df['prof_operations'] != 0), df['prof_operations'], df['profit']))
    #if all else fails, impute ebitda based on ratio of total assets
    df['ebitda'] = df['ebitda'].fillna(df['asst_tot'] * 0.05)
    df['cash_and_equiv'] = df['cash_and_equiv'].fillna(df['asst_tot'] * 0.05)
    df['profit'] = df['profit'].fillna(0.01)

    #do roe
    df['roe'] = df['profit'] / df['eqty_tot']
    df['roe'] = df['roe'].fillna(0)

    # Define if we have ebitda and operating rev is 0, set to ebitda
    # df['rev_operating'] = df.apply(lambda row: 
    #                              row['ebitda'] if row['rev_operating'] == 0 and row['ebitda'] > 0 
    #                              else 1 if row['rev_operating'] == 0 and row['ebitda'] <= 0
    #                              else row['rev_operating'], axis=1)
    # df['rev_operating'] = df['rev_operating'].fillna(df['ebitda']).fillna(1)
    df['rev_operating'] = np.where((df['rev_operating'] == 0) & (df['ebitda'] > 0), df['ebitda'],
         np.where((df['rev_operating'] == 0) & (df['ebitda'] <= 0), 1, df['rev_operating']))
    df['rev_operating'] = df['rev_operating'].fillna(df['ebitda']).fillna(1)

    df['cf_operations'] = df['cf_operations'].fillna(df['rev_operating'])


    #make unique city id for 
    df['HQ_city'] = df['HQ_city'].fillna(1000.0)

    #set AR to 0 if not present
    df['AR'] = df['AR'].fillna(0)
    #set asst_tot to eqty_tot if not present, else set to 0
    df['asst_tot'] = np.where(df['asst_tot'].isna() | (df['asst_tot'] == 0), df['eqty_tot'], df['asst_tot'])
    df['asst_tot'] = df['asst_tot'].fillna(0)
    df['eqty_tot'] = df['eqty_tot'].fillna(0)
    #set debt_st or debt_lt to 0 if na
    df['liab_lt'] = df['liab_lt'].fillna(0)
    df['debt_st'] = df['debt_st'].fillna(0)
    df['debt_lt'] = df['debt_lt'].fillna(0)


    
    return df

def add_sector_group(df, sector_col='ateco_sector', new_col='sector_group'):
    """
    Adds a 'sector_group' column to the DataFrame based on grouped sector mappings for sector codes.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the sector codes.
    - sector_col (str): Column name for sector codes in the DataFrame (default is 'sector_code').
    - new_col (str): Column name for the new sector group mapping (default is 'sector_group').

    Returns:
    - pd.DataFrame: DataFrame with an additional 'sector_group' column.
    """
    # Define grouped sector mappings
    grouped_sectors = {
        'Construction and Real Estate': [41.0, 42.0, 43.0, 68.0, 71.0, 81.0],
        'Materials and Fabrication': [7.0, 8.0, 16.0, 17.0, 19.0, 20.0, 22.0, 23.0, 24.0, 25.0, 37.0, 38.0, 39.0, 46.0],
        'Machinery and Equipment': [26.0, 27.0, 28.0, 31.0, 33.0, 52.0],
        'Automobiles and Transport': [29.0, 30.0, 45.0, 49.0, 50.0, 51.0, 53.0],
        'Consumer Products and Retail': [10.0, 11.0, 12.0, 13.0, 14.0, 47.0, 55.0, 56.0],
        'Technology, Media, and Telecommunications (TMT)': [58.0, 59.0, 60.0, 61.0, 62.0, 63.0],
        'Energy and Utilities': [5.0, 6.0, 35.0, 36.0],
        'Healthcare and Social Services': [21.0, 86.0, 87.0, 88.0],
        'Services and Professional Activities': [69.0, 70.0, 72.0, 73.0, 74.0, 77.0, 78.0, 79.0],
        'Mixed-Industry Sectors': [1.0, 2.0, 3.0, 90.0, 91.0, 92.0, 93.0, 94.0, 99.0]
    }
    
    # Reverse mapping from sector codes to group names
    sector_code_to_group = {code: group for group, codes in grouped_sectors.items() for code in codes}
    
    # Map the sector codes in the DataFrame to sector groups
    df[new_col] = df[sector_col].map(sector_code_to_group).fillna('Unknown')  # Assign 'Unknown' if no match found
    
    return df

def add_region_group(df, province_col='HQ_city', new_col='regional_code'):
    """
    Adds a 'regional_code' column to the DataFrame based on grouped region mappings for province codes.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the province codes.
    - province_col (str): Column name for province codes in the DataFrame.
    - new_col (str): Column name for the new region group mapping (default is 'regional_code').

    Returns:
    - pd.DataFrame: DataFrame with an additional 'regional_code' column.
    """
    # Define grouped region mappings
    grouped_regions = {
        '1': [1, 2, 3, 4, 5, 6, 96, 103],
        '2': [7],
        '3': [108, 98, 20, 19, 18, 17, 15, 16, 14, 13, 12, 97],
        '4': [21, 22],
        '5': [28, 29, 27, 26, 25, 24, 23],
        '6': [30, 25, 31, 32, 93],
        '7': [11, 10, 8, 9],
        '8': [39, 41, 99, 40, 38, 33, 36, 35, 34, 37],
        '9': [52, 100, 53, 50, 51, 48, 49, 47, 46, 45],
        '10': [54, 55],
        '11': [41, 42, 43, 44, 109],
        '12': [60, 58, 59, 56, 57],
        '13': [66, 67, 68, 69],
        '14': [70, 94],
        '15': [61, 62, 63, 64, 65],
        '16': [75, 110, 74, 71, 72, 73],
        '17': [76, 77],
        '18': [78, 79, 80, 101, 102],
        '19': [89, 88, 87, 86, 84, 83, 82, 81, 85],
        '20': [106, 104, 90, 91, 105, 92, 95, 107]
    }

    
    # Create a reverse mapping from province code to region code
    province_to_region_mapping = {province: region for region, provinces in grouped_regions.items() for province in provinces}
    
    # Map the province codes in the DataFrame to region codes
    df[new_col] = df[province_col].map(province_to_region_mapping).fillna('Unknown')  # Assign 'Unknown' if no match found
    
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
    #doe roe too
    df, preproc_params = make_quantiles(df, field='roe', custom_bins=custom_bins, num_quantiles=quantiles, new=new, preproc_params=preproc_params)
    
    # Calculate net income growth by ID and sort by statement date
    df['net_income'] = df['profit']
    df = create_growth_features(df, historical_df = historical_df, id_col='id', date_col='stmt_date', field='net_income', new=new)
    df, preproc_params = make_quantiles(df, field='net_income_growth', custom_bins=custom_bins, num_quantiles=quantiles, new=new, preproc_params=preproc_params)

    # Calculate Quick Ratio Version 2
    df['quick_ratio_v2'] = np.where(df['debt_st'] == 0, 100, (df['cash_and_equiv'] + df['AR']) / df['debt_st'])
    #fill with median, mean is too high, skewed
    df['quick_ratio_v2'] = df['quick_ratio_v2'].fillna(df['asst_tot'] * 0.8)
    df, preproc_params = make_quantiles(df, field='quick_ratio_v2', num_quantiles=quantiles, new=new, preproc_params=preproc_params)

    # Calculate sales growth by ID and sort by statement date
    df['sales'] = df['rev_operating']
    df = create_growth_features(df, historical_df = historical_df, id_col='id', date_col='stmt_date', field='sales', new=new)
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

    #handle sector data
    df = add_sector_group(df)
    df, preproc_params = calculate_conditional_pd_for_categorical(df, field='ateco_sector', default_col='default' ,new=new, preproc_params=preproc_params)
    df, preproc_params = calculate_conditional_pd_for_categorical(df, field='sector_group', default_col='default',new=new, preproc_params=preproc_params)
    
    #do cities
    df = add_region_group(df)
    df, preproc_params = calculate_conditional_pd_for_categorical(df, field='regional_code', default_col='default', new=new, preproc_params=preproc_params)

    #do cfo
    df['liab_st'] = df['liab_tot'] - df['liab_lt']
    #set to positive, can't have negative liabs...
    df['liab_st'] = df['liab_st'].abs()
    df['cfo'] = np.where(df['liab_st'] == 0, 100, df['cf_operations'] / df['liab_st'])
    # df['cfo'] = df.apply(lambda row: 100 if row['liab_st'] == 0 else row['cf_operations'] / row['liab_st'], axis=1)
    df, preproc_params = make_quantiles(df, field='cfo', num_quantiles=quantiles, new=new, preproc_params=preproc_params)

    #do legal_struct
    df, preproc_params = calculate_conditional_pd_for_categorical(df, field='legal_struct', default_col='default',new=new,preproc_params=preproc_params)



    return df, preproc_params

