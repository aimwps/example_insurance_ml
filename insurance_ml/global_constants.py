TRAINING_STATUS = (("training", "training"),
                    ("complete", "complete"))


ML_MODEL_OPTIONS = (
                    ("Decision Tree","Decision Tree"),
                    ("Extra trees", "Extra trees"),
                    ("Random Forest", "Random Forest"),
                    ("HGB", "HGB"),
                    ("Cat Boost", "Cat Boost"),
                    ("GBM", "GBM"),
                    ("Light GBM", "Light GBM"),
                    ("Ada Boost","Ada Boost"),
                    ("Super Vector Machine", "Super Vector Machine"),
                    ("SGD", "SGD"),
                    ("KNN", "KNN"),
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
