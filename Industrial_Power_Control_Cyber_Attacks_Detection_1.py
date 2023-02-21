# Imports, settings and dataset view
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import FunctionTransformer
from collections import Counter
from sklearn.metrics import make_scorer, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import pickle

# Set pandas to show all columns when you print a dataframe
pd.set_option('display.max_columns', None)

# Global settings for dataset choosing
dataset = ["binaryAllNaturalPlusNormalVsAttacks", "multiclass", "triple"]
number = [n for n in range(1, 15)]
index = 0
model_list = []
result_list = []
def vectorize_df(df):
    df_numeric = df.select_dtypes(include=[np.number])
    # Perform label encoder on marked column
    df['marker'] = le.fit_transform(df['marker'])
    for column in df_numeric.columns:
        if column == 'marker':
            continue
        column_data = df_numeric[column]
        # To avoid Input X contains infinity or a value too large for dtype('float64') error we replace them with float.max
        column_data = column_data.replace([np.inf, -np.inf], np.finfo(np.float64).max)
        # Check if the data is normally distributed
        if column_data.skew() < 0.5:
            df_numeric[column] = ss.fit_transform(column_data.values.reshape(-1,1))
        # Check if the data has extreme outliers
        elif column_data.quantile(0.25) < -3 or column_data.quantile(0.75) > 3:
            df_numeric[column] = rs.fit_transform(column_data.values.reshape(-1,1))
        # Check if the data has a Gaussian-like distribution
        elif 0.5 < column_data.skew() < 1:
            df_numeric[column] = lt.fit_transform(column_data.values.reshape(-1,1))
        # Check if the data can be transformed into a Gaussian-like distribution
        elif column_data.skew() > 1:
            df_numeric[column] = qt.fit_transform(column_data.values.reshape(-1,1))
        else:
            df_numeric[column] = mms.fit_transform(column_data.values.reshape(-1,1))
            df[df_numeric.columns] = df_numeric
    return df

def create_grid_search(model, params):
    # Create a grid search object which is used to find the best hyperparameters for the model
    from sklearn.model_selection import GridSearchCV
    return GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, verbose=3, cv=3, scoring='accuracy', return_train_score=True)

def show(model, X_test, y_test):
    # We print our results
    sns.set(rc={'figure.figsize': (15, 8)})
    predictions = model.predict(X_test)
    true_labels = y_test
    cf_matrix = confusion_matrix(true_labels, predictions)
    model_report = classification_report(true_labels, predictions, digits=5)
    heatmap = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))

    # The heatmap is cool but this is the most important result
    print(model_report)
    return model_report

for number_index in range(15):
    print("Currently working on "+str(dataset[index])+"/data"+str(number[number_index])+".csv")
    models = []
    results = []
    relevant = "./Class/"+str(dataset[index])+"/data"+str(number[number_index])+".csv"
    with open(relevant, 'rb') as file:
        df = pd.read_csv(file)

    for column in df.columns[df.isna().any()].tolist():
        df[column] = df[column].fillna(0.0)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.reset_index()

    # LabelEncoder encodes labels with a value between 0 and n_classes-1
    le = LabelEncoder()
    # StandardScaler scales values by subtracting the mean and dividing by the standard deviation
    ss = StandardScaler()
    # QuantileTransformer transforms features using quantiles information
    qt = QuantileTransformer()
    # RobustScaler scales values by subtracting the median and dividing by the interquartile range
    rs = RobustScaler()
    # MinMaxScaler scales values between 0 and 1
    mms = MinMaxScaler()
    # LogTransformer transforms features by taking the natural logarithm
    lt = FunctionTransformer(np.log1p)
    # Preprocessing
    df = vectorize_df(df)

    # Choose features for the model
    features_list = df.columns.to_list()
    features_list.remove('marker')
    features_list.remove('index')
    
    # Train test split
    X = df[features_list]
    y = np.stack(df['marker'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    counter = Counter(y)

    # Feature selection

    from sklearn.feature_selection import RFECV
    from sklearn.model_selection import StratifiedKFold

    # Create the RFE object and compute a cross-validated score.
    # The "accuracy" scoring is proportional to the number of correct classifications
    recall_scorer = make_scorer(recall_score, pos_label=1, average='macro')
    rfecv = RFECV(estimator=RandomForestClassifier(n_estimators=750,criterion= "entropy",max_depth= 20, min_samples_split= 2, random_state=43), step=1, cv=StratifiedKFold(2), scoring='accuracy', verbose=1, n_jobs=-1)
    #rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=StratifiedKFold(2), scoring='accuracy', verbose=1, n_jobs=-1)
    rfecv.fit(X_train, y_train)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    fig.savefig('1/m'+str(number_index)+'_rfecv.png')

    X_train = rfecv.transform(X_train)
    X_test = rfecv.transform(X_test)

    # Random Forest Classifier
    rf = RandomForestClassifier()
    rf_params = {
        "n_estimators": [150, 250, 750],
        "criterion": ["gini", "entropy"],
        "max_depth": [20],
        "min_samples_split": [2],
        "random_state": [43],
    }
    rf_grid = create_grid_search(rf, rf_params)
    rf_grid.fit(X_train, y_train)
    rf = rf_grid.best_estimator_
    pickle.dump(rf, open('1/m'+str(number_index)+'_rfc_grid.pkl', 'wb'))
    results.append(show(rf, X_test, y_test))

    # Random Forest Classifier + AdaBoost
    rf_ada = AdaBoostClassifier(base_estimator=rf)
    rf_ada_params = {
        'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    }
    rf_ada_gcv = create_grid_search(rf_ada, rf_ada_params)
    rf_ada_gcv.fit(X_train, y_train)

    rf_ada = rf_ada_gcv.best_estimator_

    # Save the model
    pickle.dump(rf_ada, open('1/m'+str(number_index)+'_rf_ada_gcv.pkl', 'wb'))

    results.append(show(rf_ada, X_test, y_test))

    # K Nearest Neighbors
    knn = KNeighborsClassifier()
    knn_params = {
        "n_neighbors": [3],
        "weights": ["distance"],
        "algorithm": ["auto"],
        "leaf_size": [10],
        "p": [1]
    }
    knn_grid = create_grid_search(knn, knn_params)
    knn_grid.fit(X_train, y_train)
    knn = knn_grid.best_estimator_
    pickle.dump(knn, open('1/m'+str(number_index)+'_knn_grid.pkl', 'wb'))
    show(knn,results, X_test, y_test)

    # Neural Network Classifier
    nn = MLPClassifier()
    nn_params = {
        "hidden_layer_sizes": [(100, 100, 100, 100, 100)],
        "activation": ["tanh"],
        "solver": ["adam"],
        "alpha": [0.01],
        "learning_rate": ["adaptive"],
    }   
    nn_grid = create_grid_search(nn, nn_params)
    nn_grid.fit(X_train, y_train)
    nn = nn_grid.best_estimator_
    pickle.dump(nn, open('1/m'+str(number_index)+'_nn_grid.pkl', 'wb'))
    results.append(show(nn, X_test, y_test))

    # Stacking Classifier ( Combining all the models )
    from sklearn.ensemble import StackingClassifier
    sc = StackingClassifier(estimators=[('rf_ada', rf_ada), ('knn', knn), ('nn', nn)], final_estimator=LogisticRegression())
    sc.fit(X_train, y_train)
    pickle.dump(sc, open('1/m'+str(number_index)+'_sc_grid.pkl', 'wb'))
    results.append(show(sc, X_test, y_test))

    model_list.append(models)
    result_list.append(results)

# Draw a graph which compares the performance of all the models on every one of the 15 datasets (not a function)
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title('Model Comparison')
ax.set_ylabel('Accuracy')
ax.set_xlabel('Model')
ax.boxplot(result_list)
ax.set_xticklabels(model_list)
plt.show()
# Save plt as a png file
fig.savefig('1/Model_Comparison.png')

# Save model_list and result_list
import pickle
pickle.dump(model_list, open('1/model_list.pkl', 'wb'))
pickle.dump(result_list, open('1/result_list.pkl', 'wb'))



