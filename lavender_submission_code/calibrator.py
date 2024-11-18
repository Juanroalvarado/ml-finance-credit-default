
# avg population rate based on (2010,2011,2012)= (2.4 + 2.2 + 3.5)/3= 2.7

def calibrator(pred,sample_rate=0.0126,population_rate=0.027):

     # Validate inputs
    # if not isinstance(pred, (int, float)):
    #     raise ValueError("Prediction value (pred) must be a numeric type.")
    # if not isinstance(sample_rate, (int, float)) or not isinstance(population_rate, (int, float)):
    #     raise ValueError("Sample rate and population rate must be numeric types.")
    
     # Handle edge cases
    # if sample_rate <= 0 or population_rate <= 0:
    #     raise ValueError("Sample rate and population rate must be positive values.")
    
    # if pred < 0 or pred > 1:
    #     raise ValueError("Prediction (pred) must be in the range [0, 1] for probability calibration.")
    
    # if sample_rate == population_rate:
    #     print("Warning: Sample rate equals population rate. Calibration may be ineffective.")
    
    # Calculate the denominator
    denominator = (sample_rate - pred * sample_rate + pred * population_rate - sample_rate * population_rate)
    
    # Avoid division by zero
    # if denominator == 0:
    #     raise ZeroDivisionError("Denominator became zero. Check values of pred, sample_rate, and population_rate.")
    
    # Calculate the calibrated prediction
    pred_calibrated = population_rate * ((pred - pred * sample_rate) / denominator)
    
    # Ensure calibrated value is within the expected range [0, 1]
    # pred_calibrated = max(0, min(pred_calibrated, 1))
    
    return pred_calibrated

  
