from sklearn.tree           import DecisionTreeRegressor
from sklearn.ensemble       import ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm            import SVR
from sklearn.linear_model   import Ridge, Lasso, SGDRegressor, BayesianRidge
from sklearn.neighbors      import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.experimental   import enable_hist_gradient_boosting
from sklearn.ensemble       import HistGradientBoostingRegressor
from catboost               import CatBoostRegressor
from lightgbm               import LGBMRegressor
# YOUR CODE HERE


mult_num_var = pipeline.Pipeline(steps=[
    ('STANDARDSCALER',  preprocessing.StandardScaler()),
    ])

mult_cat_var = pipeline.Pipeline(steps=[
    ('ONEHOTENCODER', preprocessing.OneHotEncoder(handle_unknown='ignore'))
    ])

mult_prepro = compose.ColumnTransformer(transformers=[
    ("MULT_NUM_VAR", mult_num_var, num_vars),
    ("MULT_CAT_VAR", mult_cat_var, cat_vars),
    ], remainder='passthrough')

tree_cat_var = pipeline.Pipeline(steps=[
    ('ONEHOTENCODER', preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])

tree_prepro = compose.ColumnTransformer(transformers=[
    #("TREE_NUM_VAR", tree_num_var, num_vars),
    ("TREE_CAT_VAR", tree_cat_var, cat_vars),
    ], remainder='passthrough')



tree_classifiers = {
    "Decision_tree_regressor": DecisionTreeRegressor(),
    "AdaBoost_regressor": AdaBoostRegressor(),
    "Extra_trees_regressor": ExtraTreesRegressor(),
    #"Random_forest_regressor": RandomForestRegressor(), # Takes 55 seconds
    #"GBM_regressor": GradientBoostingRegressor(), Takes forever
    "HGB_regressor": HistGradientBoostingRegressor(),
    "CATBoost_regressor": CatBoostRegressor(verbose=0),
    "lightgbm_regressor": LGBMRegressor(),
        }

mult_classifiers = {
    #"Linear_regression": LinearRegression(), ### Dont use results were awful
    "Ridge_regressor": Ridge(),
    #"SVM_regressor": SVR(), # Takes 150  seconds
    "MLP_regressor": MLPRegressor(),
    "SGD_regressor": SGDRegressor(),
    "KNN_regressor": KNeighborsRegressor(),
    "BR_regressor" : BayesianRidge(),
    #"RNN_regressor": RadiusNeighborsRegressor(), # Predicts NaN's :S
        }
tree_pipes = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}
mult_pipes = {name: pipeline.make_pipeline(mult_prepro, model) for name, model in mult_classifiers.items()}
all_pipelines = {**tree_pipes,**mult_pipes}
x_train, x_valid, y_train, y_valid = model_selection.train_test_split(x, y, test_size=0.4, random_state=0)

results = pd.DataFrame({'Model': [], 'MSE': [], 'RMSE': [], 'Time': []})

for name, pipe in all_pipelines.items():
    print(f"Working on {name}....")
    start_time = time.time()
    pipe.fit(x_train, y_train)
    preds = pipe.predict(x_valid)
    total_time = time.time() - start_time

    results = results.append({"Model": name,
                              "MSE":   metrics.mean_squared_error(y_valid, preds, squared=True),
                              "RMSE":  metrics.mean_squared_error(y_valid, preds, squared=False),
                              "Time":  total_time},
                              ignore_index=True)

    results_ord = results.sort_values(by=['RMSE'], ascending=True, ignore_index=True)
    results_ord.index += 1
    clear_output()
    display(results_ord.style.bar(subset=['MSE', 'RMSE'], vmin=0, color='#5fba7d'))
print("Finished")
