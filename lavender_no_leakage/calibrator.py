
# avg population rate based on (2010,2011,2012)= (2.4 + 2.2 + 3.5)/3= 2.7

def calibrator(pred,sample_rate=0.0126,population_rate=0.027):

    # Calculate the denominator
    denominator = (sample_rate - pred * sample_rate + pred * population_rate - sample_rate * population_rate)
    
    # Calculate the calibrated prediction
    pred_calibrated = population_rate * ((pred - pred * sample_rate) / denominator)
    
    return pred_calibrated

  
