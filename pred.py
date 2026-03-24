import numpy as np
import pandas as pd

def predict(row):
    
    return None


def predict_all(filename):
    """
    Make predictions for the data in filename
    """

    # Read the file containing the test data
    data = pd.read_csv(filename)
    
    predictions = []
    
    for idx, row in data.iterrows():
        pred = predict(row)
        predictions.append(pred)
        
    return predictions