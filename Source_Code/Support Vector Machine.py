
import sys
import csv
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# %matplotlib inline (Uncomment if using Jupyter Notebook)
import os
def read_datasets():

    
    # Ensure the files exist before reading
    user_file = r"Fake-Profile-Detection-using-ML-master\data\users.csv"

    fake_user_file =r"Fake-Profile-Detection-using-ML-master\data\fusers.csv"

    
    if not os.path.exists(user_file) or not os.path.exists(fake_user_file):
        raise FileNotFoundError("One or both dataset files are missing.")
    
    # Read datasets
    genuine_users = pd.read_csv(user_file)
    fake_users = pd.read_csv(fake_user_file)
    
    # Combine the datasets
    X = pd.concat([genuine_users, fake_users], ignore_index=True)
    
    # Assign labels: 1 for genuine users, 0 for fake users
    y = pd.Series([1] * len(genuine_users) + [0] * len(fake_users), name="label")
    
    # Shuffle data to prevent order bias
    X, y = shuffle(X, y, random_state=42)
    
    return X, y
import gender_guesser.detector as gender
import pandas as pd
import numpy as np


def predict_sex(name):
    """
    Predicts gender based on first names using `gender-guesser`.
    """
    sex_predictor = gender.Detector(case_sensitive=False)
    first_name = name.str.split(' ').str.get(0)
    
    # Ensure 'first_name' is not empty before prediction
    sex = first_name.apply(lambda n: sex_predictor.get_gender(n) if isinstance(n, str) and n else 'unknown')  # Handle empty strings or NaNs

    sex_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'mostly_male': 1, 'male': 2}
    sex_code = sex.map(sex_dict).fillna(0).astype(int)  # Handle potential NaN values from mapping
    return sex_code

def extract_features(x):
    lang_list = list(enumerate(np.unique(x['lang'].dropna())))  # Drop NaN values before mapping
    lang_dict = {name: i for i, name in lang_list}
    
    x.loc[:, 'lang_code'] = x['lang'].map(lang_dict)
    x.loc[:, 'sex_code'] = predict_sex(x['name'])

    # Fill missing values with a default code (e.g., -1)
    x['lang_code'].fillna(-1, inplace=True)
    x['sex_code'].fillna(0, inplace=True)

    # Convert to integer after handling NaN values
    x['lang_code'] = x['lang_code'].astype(int)
    x['sex_code'] = x['sex_code'].astype(int)

    feature_columns_to_use = [
        'statuses_count', 'followers_count', 'friends_count',
        'favourites_count', 'listed_count', 'sex_code', 'lang_code'
    ]

    x = x.loc[:, feature_columns_to_use]
    return x
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    """
    Plots the learning curve of an estimator.

    Parameters:
    - estimator: The model (e.g., a scikit-learn classifier or regressor).
    - title: Title of the plot.
    - X: Feature matrix.
    - y: Target vector.
    - ylim: Tuple (ymin, ymax) to set y-axis limits.
    - cv: Cross-validation strategy (integer or CV splitter).
    - n_jobs: Number of parallel jobs (-1 for all cores, default is None).
    - train_sizes: Proportion of data used for training.

    Returns:
    - Matplotlib plot object.
    """

    plt.figure(figsize=(8, 6))  # Increased figure size for clarity
    plt.title(title, fontsize=14)
    plt.xlabel("Training Examples", fontsize=12)
    plt.ylabel("Score", fontsize=12)

    if ylim is not None:
        plt.ylim(*ylim)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid(True, linestyle="--", alpha=0.7)

    # Plot shaded areas for standard deviation
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="red")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="green")

    # Plot mean scores
    plt.plot(train_sizes, train_scores_mean, 'o-', color="red", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="green", label="Cross-validation Score")

    plt.legend(loc="best", fontsize=12)
    plt.show()
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names=['Fake','Genuine']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
   
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_test, y_pred):
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_pred)
    print("False Positive rate:", false_positive_rate)
    print("True Positive rate:", true_positive_rate)
    
    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure(figsize=(8, 6))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')  # Diagonal line for random guessing
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def plot_learning_curve(estimator, title, X, y, cv):
    """Placeholder function for learning curve plotting."""
    pass  # Implement this function if needed

def train(X_train, y_train, X_test):
    """ Trains and predicts dataset with an SVM classifier """
    
    # Standardizing the features (avoid data leakage)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameter tuning
    Cs = 10.0 ** np.arange(-2, 3, 0.5)
    gammas = 10.0 ** np.arange(-2, 3, 0.5)
    param_grid = [{'gamma': gammas, 'C': Cs}]

    # Stratified K-Fold for cross-validation
    cvk = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifier = SVC(kernel='rbf')
    clf = GridSearchCV(classifier, param_grid=param_grid, cv=cvk, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("The best classifier is:", clf.best_estimator_)

    # Refit the best estimator
    best_clf = clf.best_estimator_
    best_clf.fit(X_train, y_train)

    # Cross-validation score
    scores = cross_val_score(best_clf, X_train, y_train, cv=cvk)
    print("Cross-validation scores:", scores)
    print('Estimated score: {:.5f} (+/- {:.5f})'.format(scores.mean(), scores.std() / 2))

    # Plot learning curve (Placeholder function)
    title = 'Learning Curves (SVM, RBF kernel, γ={:.6f})'.format(best_clf.gamma)
    plot_learning_curve(best_clf, title, X_train, y_train, cv=cvk)
    plt.show()

    # Predict class
    y_pred = best_clf.predict(X_test)
    return y_pred

print("Reading datasets...\n")
x, y = read_datasets()

print("Extracting features...\n")
x = extract_features(x)
print (x.columns)
print (x.describe())

print ("spliting datasets in train and test dataset...\n")
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.20, random_state=44)

print ("training datasets.......\n")
y_test = train(X_train,y_train,X_test)
print("Training datasets...\n")
y_pred = train(X_train, y_train, X_test)
print("Classification Accuracy on Test dataset:", accuracy_score(y_test, y_pred))

cm=confusion_matrix(y_test, y_pred)
print('Confusion matrix, without normalization')
print(cm)
plot_confusion_matrix(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')   
print(classification_report(y_test, y_pred, target_names=['Fake','Genuine']))  
plot_roc_curve(y_test, y_pred)

    