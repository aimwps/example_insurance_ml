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
