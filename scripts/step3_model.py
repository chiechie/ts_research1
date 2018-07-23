# -*- coding: utf-8 -*-
# Standard library
from os import listdir
import time
from os.path import join
# Third Party Library
import numpy as np
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
# My Library
from common.path_helper import saveDF, readDF, split_dir
from settings import Config_json, get_user_data_dir

config_json = Config_json()
root_dir = get_user_data_dir()
input_dir = join(root_dir, config_json.get_config("STEP1_DATA_SUBDIR"))
output_dir = join(root_dir, config_json.get_config("STEP2_DATA_SUBDIR"))

DELTA_T2_POINTS = config_json.get_config("DELTA_T2_POINTS")
TRAIN_POINTS = config_json.get_config("TRAIN_POINTS")
TICK_SEC = config_json.get_config("TICK_SEC")
PREDICT_LENGTH_POINTS = config_json.get_config("PREDICT_LENGTH_POINTS")
TRAIN_UPDATE_SEC = config_json.get_config("TRAIN_UPDATE_SEC")
TRAIN_UPDATE_POINTS = TRAIN_UPDATE_SEC // TICK_SEC


def fit_sequence(X, Y):
    Y_hat = np.zeros(Y.shape[0])
    id_train = 1
    start_idx = TRAIN_POINTS + DELTA_T2_POINTS
    for end_idx in range(start_idx, X.shape[0] - 1, TRAIN_UPDATE_POINTS):
        print("#"*50)
        print("%s%s%s%s%s" % ("#"*20, "begin ", id_train, "th training", "#"*20))
        begin = time.time()
        trainX = X.copy()[end_idx-TRAIN_POINTS:end_idx]
        trainY = Y.copy()[end_idx-TRAIN_POINTS:end_idx]
        print("train's shape", trainX.shape)
        trainX = RobustScaler(with_centering=False, quantile_range=(10, 90)).fit_transform(trainX)
#         trainX = PCA(n_components=5).fit_transform(trainX)
        classif = OneVsRestClassifier(SVC(class_weight="balanced", kernel="rbf"))
#         classif = OneVsRestClassifier(clf_dict["model"])
        classif.fit(trainX, trainY)
        print("%s%s%s" % ("#"*10, "confusion_matrix for train set",
                        "#"*10))
        print(confusion_matrix(trainY, classif.predict(trainX)))
        print("%s%s%s" % ("#"*10, "classification_report for train set",
                        "#"*10))
        print(classification_report(trainY, classif.predict(trainX)))

        predict_len = Y_hat.size - (end_idx - TRAIN_POINTS)
        Y_hat[end_idx-TRAIN_POINTS:] = classif.predict(X.iloc[end_idx-TRAIN_POINTS:, :].values.reshape(predict_len, -1))
        print("%s%s%s" % ("#"*10, "confusion_matrix for test set",
                        "#"*10))
        print(confusion_matrix(Y.copy()[TRAIN_POINTS:TRAIN_POINTS + 15],
                               classif.predict(X.copy()[TRAIN_POINTS:TRAIN_POINTS + PREDICT_LENGTH_POINTS])))
        print("%s%s%s" % ("#"*10, "classification_report for test set",
                        "#"*10))
        print(classification_report(Y.copy()[TRAIN_POINTS:TRAIN_POINTS + PREDICT_LENGTH_POINTS],
                               classif.predict(X.copy()[TRAIN_POINTS:TRAIN_POINTS + PREDICT_LENGTH_POINTS])))

        end = time.time()

        print("error", Y_hat[end_idx], Y[end_idx], Y_hat[end_idx] - Y[end_idx])
        print("cost %s ms" % (int((end - begin) * 1000)))
        id_train += 1
    return Y_hat


RNG = np.random.RandomState(42)
N_jobs = 10
CV_DEBUG = ShuffleSplit(n_splits=2, test_size=0.5, random_state=RNG)
CV = ShuffleSplit(n_splits=5, test_size=0.2, random_state=RNG)

ensemble_clfs_params = {'n_estimators': [100],
                              'max_depth': [3, 5],
                              'min_samples_split': [15],
                              'min_samples_leaf': [10, ],
                              'max_features': ['auto'],
                              'loss': ['deviance', ],
                              }

ensemble_clfs_params_debug = {'n_estimators': [10],
                              'max_depth': [3, ],
                              'min_samples_split': [15, ],
                              'min_samples_leaf': [10, ],
                              'max_features': ['auto'],
                              'loss': ['deviance', ],
                              }

clf_dict = {
    "model": ensemble.GradientBoostingClassifier(random_state=RNG, verbose=1),
    "params": ensemble_clfs_params,
    "params_debug": ensemble_clfs_params_debug
}


def model_train(X, Y):
    CLF = GridSearchCV(OneVsRestClassifier(clf_dict["model"]),
                       # clf_dict["params"],
                       {},
                       scoring=scoring_clf,
                       verbose=0, cv=CV, n_jobs=N_jobs)
    Y = np.array(Y.values, dtype=int).squeeze()
    CLF.fit(X, Y.reshape(-1, 1))
    print 'best params:\n', CLF.best_params_
    mean_scores = np.array(CLF.cv_results_['mean_test_score'])
    print 'mean score', mean_scores
    print 'best score', CLF.best_score_
    print 'worst score', np.min(mean_scores)
    clf = CLF.best_estimator_
    return clf, X.columns


def scoring_clf(clf, x, y):
    pred_prob = model_predict(clf, x)
    assert pred_prob.size == y.size
    average_precision = average_precision_score(y, pred_prob,)
    return average_precision


def model_predict(clf, X):
    return clf.predict_proba(X)[:, 1]


def heldout_score(clf, X, Y):
    """compute deviance scores on ``X_test`` and ``y_test``. """
    n_estimators, n_classes = clf.estimators_.shape
    score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X)):
        score[i] = clf.loss_(Y, y_pred)
    return score


if __name__ == "__main__":
    df_list = [join(input_dir, i) for i in listdir(input_dir) if ".csv" in i]
    for df_name in df_list[:1]:
        df = readDF(df_name)
        Y = df["label"]
        X = df.copy()
        del X["label"]
        Y_hat = fit_sequence(X.copy(), Y.copy())
        from sklearn.metrics.classification import classification_report
        print(df_name)
        print(classification_report(Y[TICK_SEC + DELTA_T2_POINTS:], Y_hat[TICK_SEC + DELTA_T2_POINTS:]))
        print("%s%s%s" % ("#"*10, "confusion_matrix for train set",
                        "#"*10))
        print(confusion_matrix(Y[TICK_SEC + DELTA_T2_POINTS:-PREDICT_LENGTH_POINTS], Y_hat[TRAIN_POINTS + DELTA_T2_POINTS:- PREDICT_LENGTH_POINTS]))
        df["pred"] = Y_hat
        _dir, _filename = split_dir(df_name)
        out_path = join(output_dir, _filename)
        saveDF(df, out_path)

