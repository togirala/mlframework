from sklearn import ensemble
import sys

TRAINING_DATA = 'input/train_folds.csv'
TEST_DATA = 'input/test.csv'
MODEL = 'randomforest'

# FOLDS = [0,1,2,3,4]
FOLDS = [0, 1, 2]

# 0.75091


MODELS = {
    "randomforest": ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1, verbose=2),
    "extratrees": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
}