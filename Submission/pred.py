import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_PREPROCESSED_DIR = PROJECT_ROOT / "processed" / "preprocessing"

from models.common import (  # noqa: E402
    apply_feature_subset,
    classification_report,
    dense_load,
    load_feature_names,
    load_preprocessor,
    preprocess_frame,
    read_rows,
    select_feature_indices,
)
from processed.models.naive_bayes.train_naivebayes import (
    make_prediction,
    naive_bayes_map,
)



def predict_all(filename):
    """
    Make predictions for the data in filename
    """

    # Read the file containing the test data
    data = dense_load(filename)
    predictions = []
    
    # Load the pre-processed training data and feature names
    preprocessor = load_preprocessor(DEFAULT_PREPROCESSED_DIR)
    frame = read_rows(filename, preprocessor)
    x = preprocess_frame(frame, preprocessor)
    
    # Trained Naive Bayes model parameters
    alpha = 6.32
    p_c, p_x_given_c = naive_bayes_map(X_train, y_train, alpha, method = "binary")
    y_nb = make_prediction(data, p_c, p_x_given_c)
    
        
    return predictions