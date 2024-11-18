import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def calculate_marginal_effects_numpy(model, X_train, x_new, h=0.01):
    """
    Optimized version of calculate_marginal_effects using NumPy for internal operations.
    
    Parameters:
    - model: Trained machine learning model with a `predict_proba` method.
    - X_train: Training data (DataFrame) used to calculate standard deviations of features.
    - x_new: New record (DataFrame with one row) for which to calculate marginal effects.
    - h: Size of relative perturbation (default is 0.01).
    
    Returns:
    - dy_dx: Marginal effects for each feature (Series).
    """
    try:
        # Convert x_new to NumPy array and flatten
        x_new = x_new.to_numpy().flatten()
        
        # Calculate standard deviation for each feature in X_train
        std_devs = np.std(X_train.to_numpy(), axis=0)
        
        # Predict probability for the current record
        y_est = model.predict_proba(x_new.reshape(1, -1))[0][1]
        
        # Initialize array for marginal effects
        dy_dx = np.zeros_like(x_new)
        
        # Loop through each feature
        for j in range(len(x_new)):
            true_z = x_new[j]
            
            # Perturb the feature up and down
            z_down = true_z - h * std_devs[j]
            z_up = true_z + h * std_devs[j]
            
            # Create perturbed copies
            x_down = x_new.copy()
            x_up = x_new.copy()
            x_down[j] = z_down
            x_up[j] = z_up
            
            # Predict for perturbed records
            y_down = model.predict_proba(x_down.reshape(1, -1))[0][1]
            y_up = model.predict_proba(x_up.reshape(1, -1))[0][1]
            
            # Calculate dy/dx using central difference approximation
            dy_dx_down = (y_est - y_down) / (true_z - z_down) if true_z != z_down else 0
            dy_dx_up = (y_est - y_up) / (true_z - z_up) if true_z != z_up else 0
            dy_dx[j] = (dy_dx_up + dy_dx_down) / 2
        
        return pd.Series(dy_dx, index=X_train.columns)
    except Exception as e:
        print(f"An error occurred while calculating marginal effects: {e}")
        raise


def calculate_marginal_effects_for_dataset_batch(model, X_train, X_new, h=0.01, n_jobs=-1, batch_size=100):
    """
    Optimized function to calculate marginal effects for a large dataset using batch processing and parallelism.
    
    Parameters:
    - model: Trained machine learning model with a `predict_proba` method.
    - X_train: Training data (DataFrame) used to calculate standard deviations of features.
    - X_new: New records (DataFrame) for which to calculate marginal effects.
    - h: Size of relative perturbation (default is 0.01).
    - n_jobs: Number of parallel jobs to run (-1 means use all available cores).
    - batch_size: Number of rows to process in each batch.
    
    Returns:
    - results: DataFrame with marginal effects for each record and feature.
    """
    try:
        # Validate inputs
        if not isinstance(X_train, pd.DataFrame) or not isinstance(X_new, pd.DataFrame):
            raise ValueError("X_train and X_new must be pandas DataFrames.")
        
        # Split X_new into batches
        num_batches = (len(X_new) + batch_size - 1) // batch_size
        batches = [X_new.iloc[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
        
        # Function to process a single batch
        def process_batch(batch):
            results = {}
            for idx, row in batch.iterrows():
                print(f"Processing index {idx} in batch...")
                results[idx] = calculate_marginal_effects_numpy(model, X_train, row.to_frame().T, h)
            return results
        
        # Use joblib to parallelize batch processing
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(process_batch)(batch) for batch in batches
        )
        
        # Combine all batch results into a single DataFrame
        combined_results = {idx: dy_dx for batch in batch_results for idx, dy_dx in batch.items()}
        results_df = pd.DataFrame.from_dict(combined_results, orient="index")
        results_df.index = X_new.index  # Match original index
        return results_df
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
