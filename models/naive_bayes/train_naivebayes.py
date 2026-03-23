import numpy as np
from scipy import sparse
import json

def naive_bayes_map(X, t, alpha = 1.0):
    
    N, vocab_size = X.shape[0], X.shape[1]
    
    # Calculate the class probabilities p(c) for each class c via counts
    pi_0 = np.sum(t == 0) / N
    pi_1 = np.sum(t == 1) / N
    pi_2 = np.sum(t == 2) / N
    pi = np.array([pi_0, pi_1, pi_2])
    
    theta = np.zeros([vocab_size, 3])
    
    # All data points belonging to each class
    X_pom = X[t == 0]
    X_sn = X[t == 1]
    X_wlp = X[t == 2]
    
    # Total tf-idf mass for each word in the vocab for each class
    N_xi_given_0 = X_pom.sum(axis=0)
    N_xi_given_1 = X_sn.sum(axis=0)
    N_xi_given_2 = X_wlp.sum(axis=0)
    
    # Total tf-idf mass for each class
    N_0 = N_xi_given_0.sum()
    N_1 = N_xi_given_1.sum()
    N_2 = N_xi_given_2.sum()
    
    # Calculate the p(xi | c) using a multinomial prior with Laplace smoothing
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

def main():
    # Load the pre-processed data
    feature_names = json.load(open("processed\\preprocessing\\feature_names.json", "r", encoding="utf-8"))
    X_train = np.load("processed\\preprocessing\\train_X.npz")["data"]
    y_train = np.load("processed\\preprocessing\\train_y.npy")

    X_val = np.load("processed\\preprocessing\\val_X.npz")["data"]
    y_val = np.load("processed\\preprocessing\\val_y.npy")
    
    X_test = np.load("processed\\preprocessing\\test_X.npz")["data"]
    y_test = np.load("processed\\preprocessing\\test_y.npy")
    
    # Pick out vocab from the feature names
    vocab = [f.split(":")[1] for f in feature_names if isinstance(f, str) and f.startswith("text:")]

    # bag of words data matrices with tf-idf features
    X_train_bow = X_train[:, :len(vocab)]
    X_val_bow = X_val[:, :len(vocab)]
    X_test_bow = X_test[:, :len(vocab)]
    
    # Tune alpha as a hyperparameter
    alphas = np.linspace(0.01, 2.0, 20)
    accs = []
    for alpha in alphas:
        p_c, p_x_given_c = naive_bayes_map(X_train_bow, y_train, alpha)
        y_val_pred = make_prediction(X_val_bow, p_c, p_x_given_c)
        accs.append(accuracy(y_val_pred, y_val))
    
    best_alpha = alphas[np.argmax(accs)]
    
    # Train the final model with the best alpha and evaluate on the test set
    p_c, p_x_given_c = naive_bayes_map(X_train_bow, y_train, best_alpha)
    
    y_train_pred = make_prediction(X_train_bow, p_c, p_x_given_c)
    y_val_pred = make_prediction(X_val_bow, p_c, p_x_given_c)
    y_test_pred = make_prediction(X_test_bow, p_c, p_x_given_c)
    
    print(f"Best alpha: {best_alpha:.2f}")
    print(f"Training Accuracy: {accuracy(y_train_pred, y_train):.4f}")
    print(f"Validation Accuracy: {accuracy(y_val_pred, y_val):.4f}")
    print(f"Test Accuracy: {accuracy(y_test_pred, y_test):.4f}")
 
    
if __name__ == "__main__":
    
    main()