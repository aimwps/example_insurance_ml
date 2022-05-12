from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, SGDRegressor, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from catboost import CatBoostRegressor
from lightgbm  import LGBMRegressor
from xgboost  import XGBRegressor
random_state = 894
TRAINING_STATUS = (("training", "training"),
                    ("complete", "complete"))


ML_MODEL_OPTIONS = (
                    ("decisiontree","Decision Tree"),
                    ("extratrees", "Extra trees"),
                    ("randomforest", "Random Forest"),
                    ("hgb", "HGB"),
                    ("catboost", "Cat Boost"),
                    ("gbm", "GBM"),
                    ("lightgbm", "Light GBM"),
                    ("adaboost","Ada Boost"),
                    ("svm", "Super Vector Machine"),
                    ("sgd", "SGD"),
                    ("multilayerperceptron", "Multi Layer Perceptron"),
                    ("xgboost", "Extreme Gradient Boost")
                    #("KNN", "KNN"),
                )
NUM_FIELDS = (
            ("rf_age", "Age"),
            ("rf_bmi", "Body Mass Index"),
            ("rf_children", "Number of children"),
            )

CAT_FIELDS = (
            ("rf_gender", "Gender"),
            ("rf_region", "Region of residence"),
            ("rf_is_smoker", "Proposer is smoker"),
            )
MODEL_DICT = {
        "tree": {
            "decisiontree": DecisionTreeRegressor(random_state=random_state),
            "extratrees": ExtraTreesRegressor(random_state=random_state),
            "randomforest": RandomForestRegressor(random_state=random_state),
            "hgb": HistGradientBoostingRegressor(random_state=random_state),
            "catboost": CatBoostRegressor(verbose=0, random_state=random_state),
            "gbm": GradientBoostingRegressor(random_state=random_state),
            "lightgbm": LGBMRegressor(random_state=random_state),
            "xgboost": XGBRegressor(random_state=random_state)
        },
        "mult": {
            "adaboost":AdaBoostRegressor(random_state=random_state),
            "svm": SVR(),
            "sgd": SGDRegressor(random_state=random_state),
            "multilayerperceptron": MLPRegressor(random_state=random_state),
            #"KNN": KNeighborsRegressor(),
        },
    }
PARAM_GRID = [{
            "decisiontree__criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"] ,
            "extratrees__criterion":["squared_error","absolute_error",] ,
            "randomforest__criterion": ["squared_error", "absolute_error", "poisson"] ,
            "randomforest__n_estimators": [25, 50, 100, 150, 200, 250, 500] ,
            "hgb__loss":["squared_error", "absolute_error",  "poisson"] ,
            "hgb__learning_rate":[0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001],
            "hgb__min_samples_leaf":[12, 24, 36, 48],
            "catboost__depth": [6,8,10],
            "catboost__learning_rate": [0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001],
            "catboost__iterations": [25, 50, 100, 150, 200, 250, 500],
            "gbm__criterion": ["friedman_mse", "squared_error", "absolute_error"] ,
            "gbm__n_estimators":[25, 50, 100, 150, 200, 250, 500],
            "lightgbm__learning_rate": [0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001],
            "lightgbm__boosting": ["gbdt","rf", "dart", "goss"],
            "adaboost__n_estimators": [25, 50, 100, 150, 200, 250, 500] ,
            "adaboost__loss": ["linear", "square", "exponential"],
            "adaboost__learning_rate": [0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001],
            "svm__C": [0.1,1, 10, 100],
            "svm__kernel":['rbf', 'poly', 'sigmoid'],
            "svm__gamma": [1,0.1,0.01,0.001]
            # "sgd": [],
            # "multilayerperceptron" :[] ,
            }]
