import numpy as np
import json
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPROCESSED_DIR = PROJECT_ROOT / "processed" / "preprocessing"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "processed" / "models" / "naive_bayes"
DEFAULT_INFERENCE_PARAMS_PATH = DEFAULT_OUTPUT_DIR / "nb_inference_params.npz"


def naive_bayes_map(X, t, alpha, method = "binary"):
    
    N, vocab_size = X.shape[0], X.shape[1]
    
    # Calculate the class probabilities p(c) for each class c via multinomial prior  
    pi_0 = (np.sum(t == 0)+alpha) / (N + 3 * alpha)
    pi_1 = (np.sum(t == 1)+alpha) / (N + 3 * alpha)
    pi_2 = (np.sum(t == 2)+alpha) / (N + 3 * alpha)
    pi = np.array([pi_0, pi_1, pi_2])
    
    theta = np.zeros([vocab_size, 3])
    
    # All data points belonging to each class
    X_pom = X[t == 0]
    X_sn = X[t == 1]
    X_wlp = X[t == 2]
    
    if method == "binary":
        # Using binary features - 0 for absence and 1 for presence of a word in the data point
        X_pom_binary = (X_pom > 0).astype(int)
        X_sn_binary = (X_sn > 0).astype(int)
        X_wlp_binary = (X_wlp > 0).astype(int)
        
        N_0 = np.sum(t == 0)
        N_1 = np.sum(t == 1)
        N_2 = np.sum(t == 2)
    
        N_pom_binary = X_pom_binary.sum(axis=0)
        N_sn_binary = X_sn_binary.sum(axis=0)
        N_wlp_binary = X_wlp_binary.sum(axis=0)
        
        theta[:, 0] = (N_pom_binary + alpha) / (N_0 + vocab_size * alpha)
        theta[:, 1] = (N_sn_binary + alpha) / (N_1 + vocab_size * alpha)
        theta[:, 2] = (N_wlp_binary + alpha) / (N_2 + vocab_size * alpha)
        
    else:
        # Using tf-idf features
        # Total tf-idf mass for each word in the vocab for each class
        N_xi_given_0 = X_pom.sum(axis=0)
        N_xi_given_1 = X_sn.sum(axis=0)
        N_xi_given_2 = X_wlp.sum(axis=0)
        
        # Total tf-idf mass for each class
        N_0 = N_xi_given_0.sum()
        N_1 = N_xi_given_1.sum()
        N_2 = N_xi_given_2.sum()
        
        # Calculate the p(xi | c) using a multinomial prior 
        theta[:, 0] = (N_xi_given_0 + alpha) / (N_0 + vocab_size * alpha)
        theta[:, 1] = (N_xi_given_1 + alpha) / (N_1 + vocab_size * alpha)
        theta[:, 2] = (N_xi_given_2 + alpha) / (N_2 + vocab_size * alpha)

    return pi, theta

def make_prediction(X, pi, theta):
    
    log_pi = np.log(pi)
    log_theta = np.log(theta)
    
    log_p_c_given_x = X @ log_theta + log_pi
    predictions = np.argmax(log_p_c_given_x, axis=1)
    
    return predictions

def accuracy(y, t):
    return np.mean(y == t)

def tune_alpha(X_train_bow, y_train, X_val_bow, y_val, alphas, bow_type):
    # Tune alpha as a hyperparameter
    
    accs = []
    for alpha in alphas:
        p_c, p_x_given_c = naive_bayes_map(X_train_bow, y_train, alpha, method = bow_type)
        y_val_pred = make_prediction(X_val_bow, p_c, p_x_given_c)
        accs.append(accuracy(y_val_pred, y_val))
    
    best_alpha = alphas[np.argmax(accs)]
    return best_alpha

def export_inference_params(path, pi, theta, alpha, text_feature_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pi=pi.astype(np.float32),
        theta=theta.astype(np.float32),
        alpha=np.array(alpha, dtype=np.float32),
        text_feature_count=np.array(text_feature_count, dtype=np.int64),
    )

def train_nb( X_train, y_train, X_val, y_val, X_test):
    
    # Pick out vocab from the feature names
    feature_len = 2688+17 # 2688 text features + 17 categorical features

    # bag of words data matrices with tf-idf features
    X_train_bow = X_train[:, :feature_len]
    X_val_bow = X_val[:, :feature_len]
    X_test_bow = X_test[:, :feature_len]
    
    bow_type = "binary"
    # Tuning alpha as a hyperparameter using the validation set
    alphas = np.linspace(0.01, 10, 20)
    best_alpha = tune_alpha(X_train_bow, y_train, X_val_bow, y_val, alphas, bow_type)
    
    # Train the final model with the best alpha and evaluate on the test set
    p_c, p_x_given_c = naive_bayes_map(X_train_bow, y_train, best_alpha, method = bow_type)
    
    return p_c, p_x_given_c, best_alpha, bow_type, X_train_bow, X_val_bow, X_test_bow

def eval_nb(p_c, p_x_given_c, X_train_bow, y_train, X_val_bow, y_val, X_test_bow, y_test, best_alpha, bow_type):
    
    y_train_pred = make_prediction(X_train_bow, p_c, p_x_given_c)
    y_val_pred = make_prediction(X_val_bow, p_c, p_x_given_c)
    y_test_pred = make_prediction(X_test_bow, p_c, p_x_given_c)
    
    print(f"MAP Naive Bayes Classifier Results ({bow_type} features):")
    print("Prior: Multinomial Distribution")
    print(f"Best alpha: {best_alpha:.2f}")
    print(f"Training Accuracy: {accuracy(y_train_pred, y_train):.4f}")
    print(f"Validation Accuracy: {accuracy(y_val_pred, y_val):.4f}")
    print(f"Test Accuracy: {accuracy(y_test_pred, y_test):.4f}")
 
    #Compare with sklearn's MultinomialNB
    clf = MultinomialNB(alpha=best_alpha, fit_prior=True)
    clf.fit(X_train_bow, y_train)
    y_train_pred_sklearn = clf.predict(X_train_bow)
    y_val_pred_sklearn = clf.predict(X_val_bow)
    y_test_pred_sklearn = clf.predict(X_test_bow)
    print("\nSklearn MultinomialNB Classifier Results:")
    print(f"Alpha: {best_alpha:.2f}")
    print(f"Training Accuracy: {accuracy(y_train_pred_sklearn, y_train):.4f}")
    print(f"Validation Accuracy: {accuracy(y_val_pred_sklearn, y_val):.4f}")
    print(f"Test Accuracy: {accuracy(y_test_pred_sklearn, y_test):.4f}")
    
    # Calculate Precision, Recall, F1-score for each class
    from sklearn.metrics import classification_report
    print("\nClassification Report for MAP Naive Bayes:")
    print(classification_report(y_test, y_test_pred, target_names=["POM", "SN", "WLP"]))
    
if __name__ == "__main__":
    
    # Load the pre-processed data
    feature_names = json.load(open(DEFAULT_PREPROCESSED_DIR / "feature_names.json", "r", encoding="utf-8"))
    X_train = np.load(DEFAULT_PREPROCESSED_DIR / "train_X.npz")["data"]
    y_train = np.load(DEFAULT_PREPROCESSED_DIR / "train_y.npy")

    X_val = np.load(DEFAULT_PREPROCESSED_DIR / "val_X.npz")["data"]
    y_val = np.load(DEFAULT_PREPROCESSED_DIR / "val_y.npy")
    
    X_test = np.load(DEFAULT_PREPROCESSED_DIR / "test_X.npz")["data"]
    y_test = np.load(DEFAULT_PREPROCESSED_DIR / "test_y.npy")
    
    p_c, p_x_given_c, best_alpha, bow_type, X_train_bow, X_val_bow, X_test_bow = train_nb(X_train, y_train, X_val, y_val, X_test)
    eval_nb(p_c, p_x_given_c, X_train_bow, y_train, X_val_bow, y_val, X_test_bow, y_test, best_alpha, bow_type)
    export_inference_params(DEFAULT_INFERENCE_PARAMS_PATH, p_c, p_x_given_c, best_alpha, X_train_bow.shape[1])
    print(f"Saved NB inference params to: {DEFAULT_INFERENCE_PARAMS_PATH}")
