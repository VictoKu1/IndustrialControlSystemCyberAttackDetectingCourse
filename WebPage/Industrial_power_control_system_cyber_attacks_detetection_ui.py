# UI which is being called by the Dockerfile
# and allows the user to add csv files and returns a classification
# of the data, if it is a regular operation, natural event or a cyber attack.

import warnings
warnings.simplefilter("ignore", UserWarning)
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
import pickle
import sys




scenarios = {1: "Natural events (SLG faults), Fault from 10-19% on L1",
             2: "Natural events (SLG faults), Fault from 20-79% on L1",
             3: "Natural events (SLG faults), Fault from 80-90% on L1",
             4: "Natural events (SLG faults), Fault from 10-19% on L2",
             5: "Natural events (SLG faults), Fault from 20-79% on L1",
             6: "Natural events (SLG faults), Fault from 80-90% on L1",
             7: "Data Injection, Attack Sub-type (SLG fault replay), Fault from 10-19% on L1 with tripping command",
             8: "Data Injection, Attack Sub-type (SLG fault replay), Fault from 20-79% on L1 with tripping command",
             9: "Data Injection, Attack Sub-type (SLG fault replay), Fault from 80-90% on L1 with tripping command",
             10: "Data Injection, Attack Sub-type (SLG fault replay), Fault from 10-19% on L2 with tripping command",
             11: "Data Injection, Attack Sub-type (SLG fault replay), Fault from 20-79% on L2 with tripping command",
             12: "Data Injection, Attack Sub-type (SLG fault replay), Fault from 80-90% on L2 with tripping command",
             13: "Natural events (Line maintenance), Line L1 maintenance",
             14: "Natural events (Line maintenance), Line L2 maintenance",
             15: "Remote Tripping Command Injection, Attack Sub-type (Command injection against single relay), Command Injection to R1",
             16: "Remote Tripping Command Injection, Attack Sub-type (Command injection against single relay), Command Injection to R2",
             17: "Remote Tripping Command Injection, Attack Sub-type (Command injection against single relay), Command Injection to R3",
             18: "Remote Tripping Command Injection, Attack Sub-type (Command injection against single relay), Command Injection to R4",
             19: "Remote Tripping Command Injection, Attack Sub-type (Command injection against single relay), Command Injection to R1 and R2",
             20: "Remote Tripping Command Injection, Attack Sub-type (Command injection against single relay), Command Injection to R3 and R4",
             21: "Relay Setting Change, Attack Sub-type (Disabling relay function - single relay disabled & fault), Fault from 10-19% on L1 with R1 disabled & fault",
             22: "Relay Setting Change, Attack Sub-type (Disabling relay function - single relay disabled & fault), Fault from 20-90% on L1 with R1 disabled & fault",
             23: "Relay Setting Change, Attack Sub-type (Disabling relay function - single relay disabled & fault), Fault from 10-49% on L1 with R2 disabled & fault",
             24: "Relay Setting Change, Attack Sub-type (Disabling relay function - single relay disabled & fault), Fault from 50-79% on L1 with R2 disabled & fault",
             25: "Relay Setting Change, Attack Sub-type (Disabling relay function - single relay disabled & fault), Fault from 80-90% on L1 with R2 disabled & fault",
             26: "Relay Setting Change, Attack Sub-type (Disabling relay function - single relay disabled & fault), Fault from 10-19% on L2 with R3 disabled & fault",
             27: "Relay Setting Change, Attack Sub-type (Disabling relay function - single relay disabled & fault), Fault from 20-49% on L2 with R3 disabled & fault",
             28: "Relay Setting Change, Attack Sub-type (Disabling relay function - single relay disabled & fault), Fault from 50-90% on L2 with R3 disabled & fault",
             29: "Relay Setting Change, Attack Sub-type (Disabling relay function - single relay disabled & fault), Fault from 10-79% on L2 with R4 disabled & fault",
             30: "Relay Setting Change, Attack Sub-type (Disabling relay function - single relay disabled & fault), Fault from 80-90% on L2 with R4 disabled & fault",
             35: "Relay Setting Change, Attack Sub-type (Disabling relay function - two relays disabled & fault), Fault from 10-49% on L1 with R1 and R2 disabled & fault",
             36: "Relay Setting Change, Attack Sub-type (Disabling relay function - two relays disabled & fault), Fault from 50-90% on L1 with R1 and R2 disabled & fault",
             37: "Relay Setting Change, Attack Sub-type (Disabling relay function - two relays disabled & fault), Fault from 10-49% on L1 with R3 and R4 disabled & fault",
             38: "Relay Setting Change, Attack Sub-type (Disabling relay function - two relays disabled & fault), Fault from 50-90% on L1 with R3 and R4 disabled & fault",
             39: "Relay Setting Change, Attack Sub-type (Disabling relay function - two relay disabled & line maintenance), L1 maintenance with R1 and R2 disabled",
             40: "Relay Setting Change, Attack Sub-type (Disabling relay function - two relay disabled & line maintenance), L1 maintenance with R1 and R2 disabled",
             41: "No Events (Normal operation), Normal Operation load changes"}


rfecv_binary = [['R1-PA2:VH', 'R1-PM5:I', 'R2-PM5:I', 'R2-PA7:VH', 'R3-PA3:VH', 'R3-PA6:IH', 'R3-PA7:VH', 'R4-PA2:VH', 'R4-PM5:I'],
                ['R1-PA2:VH', 'R1-PA3:VH', 'R1-PA5:IH', 'R1-PM5:I', 'R2-PA3:VH', 'R2-PM3:V', 'R2-PM5:I', 'R2-PA7:VH', 'R3-PA5:IH', 'R3-PM5:I', 'R3-PA7:VH', 'R4-PM2:V', 'R4-PA5:IH', 'R4-PM5:I'],
                ['R1-PA2:VH', 'R1-PA3:VH', 'R1-PM5:I', 'R2-PA5:IH', 'R2-PM5:I', 'R3-PA1:VH', 'R3-PA2:VH', 'R3-PM2:V', 'R3-PA3:VH', 'R3-PA5:IH', 'R3-PM5:I', 'R4-PM2:V', 'R4-PA3:VH', 'R4-PM5:I'],
                ['R1-PA1:VH', 'R1-PA2:VH', 'R1-PA3:VH', 'R1-PM5:I', 'R1-PA7:VH', 'R2-PA5:IH', 'R2-PM5:I', 'R3-PM5:I', 'R4-PA2:VH', 'R4-PM2:V', 'R4-PM5:I'],
                ['R1-PA3:VH', 'R1-PA5:IH', 'R1-PM5:I', 'R1-PA7:VH', 'R2-PM1:V', 'R2-PA2:VH', 'R2-PA5:IH', 'R2-PM5:I', 'R2-PM6:I', 'R3-PA2:VH', 'R3-PA3:VH', 'R3-PA5:IH', 'R3-PM5:I', 'R4-PA1:VH', 'R4-PM2:V', 'R4-PA3:VH', 'R4-PM5:I', 'R4-PA7:VH'],
                ['R1-PA2:VH', 'R1-PA5:IH', 'R1-PM5:I', 'R2-PA2:VH', 'R2-PA3:VH', 'R2-PM3:V', 'R2-PA5:IH', 'R2-PM5:I', 'R3-PA2:VH', 'R3-PA5:IH', 'R3-PM5:I', 'R4-PA2:VH', 'R4-PM2:V', 'R4-PM5:I', 'R4-PA7:VH'],
                ['R1-PA1:VH', 'R1-PM5:I', 'R2-PA3:VH', 'R2-PM5:I', 'R3-PA3:VH', 'R4-PM5:I'],
                ['R1-PA1:VH', 'R1-PA2:VH', 'R1-PA3:VH', 'R1-PM5:I', 'R1-PA7:VH', 'R2-PA2:VH', 'R2-PA3:VH', 'R2-PM5:I', 'R2-PM7:V', 'R3-PA2:VH', 'R3-PM2:V', 'R3-PM5:I', 'R4-PA2:VH', 'R4-PM2:V', 'R4-PA3:VH', 'R4-PM5:I'],
                ['R1-PA2:VH', 'R1-PM5:I', 'R2-PA3:VH', 'R4-PA2:VH', 'R4-PM5:I'],
                ['R1-PM5:I', 'R2-PA3:VH', 'R2-PA5:IH', 'R2-PM5:I', 'R4-PA1:VH', 'R4-PA2:VH', 'R4-PM5:I', 'R4-PA7:VH'],
                ['R1-PA2:VH', 'R1-PA3:VH', 'R1-PM5:I', 'R2-PM1:V', 'R2-PA5:IH', 'R2-PM5:I', 'R3-PA7:VH', 'R4-PM2:V', 'R4-PA3:VH', 'R4-PM5:I'],
                ['R1-PA1:VH', 'R1-PA2:VH', 'R1-PA3:VH', 'R1-PM5:I', 'R1-PA7:VH', 'R2-PA5:IH', 'R2-PM5:I', 'R3-PM5:I', 'R4-PA1:VH', 'R4-PM2:V', 'R4-PM5:I'],
                ['R1-PA3:VH', 'R1-PM5:I', 'R1-PA7:VH', 'R2-PM5:I', 'R3-PA2:VH', 'R3-PM2:V', 'R3-PA3:VH', 'R3-PM5:I', 'R4-PA1:VH', 'R4-PA3:VH', 'R4-PM5:I', 'R4-PA7:VH'],
                ['R1-PA3:VH', 'R1-PM5:I', 'R2-PA7:VH', 'R4-PA3:VH', 'R4-PM5:I']]

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
            df_numeric[column] = ss.fit_transform(column_data.values.reshape(-1, 1))
        # Check if the data has extreme outliers
        elif column_data.quantile(0.25) < -3 or column_data.quantile(0.75) > 3:
            df_numeric[column] = rs.fit_transform(column_data.values.reshape(-1, 1))
        # Check if the data has a Gaussian-like distribution
        elif 0.5 < column_data.skew() < 1:
            df_numeric[column] = lt.fit_transform(column_data.values.reshape(-1, 1))
        # Check if the data can be transformed into a Gaussian-like distribution
        elif column_data.skew() > 1:
            df_numeric[column] = qt.fit_transform(column_data.values.reshape(-1, 1))
        else:
            df_numeric[column] = mms.fit_transform(column_data.values.reshape(-1, 1))
            df[df_numeric.columns] = df_numeric
    return df






def preprocess(df):
    apparent_impedance_measurements_headers_names = ['R1-PA:Z', 'R2-PA:Z', 'R3-PA:Z', 'R4-PA:Z']
    voltage_phase_angles_headers_names = ['R1-PA1:VH', 'R1-PA2:VH', 'R1-PA3:VH',
                                          'R2-PA1:VH', 'R2-PA2:VH', 'R2-PA3:VH',
                                          'R3-PA1:VH', 'R3-PA2:VH', 'R3-PA3:VH',
                                          'R4-PA1:VH', 'R4-PA2:VH', 'R4-PA3:VH']
    current_phase_angles_headers_names = ['R1-PA4:IH', 'R1-PA5:IH', 'R1-PA6:IH',
                                          'R2-PA4:IH', 'R2-PA5:IH', 'R2-PA6:IH',
                                          'R3-PA4:IH', 'R3-PA5:IH', 'R3-PA6:IH',
                                          'R4-PA4:IH', 'R4-PA5:IH', 'R4-PA6:IH']
    voltage_phase_magnitudes_headers_names = ['R1-PM1:V', 'R1-PM2:V', 'R1-PM3:V',
                                             'R2-PM1:V', 'R2-PM2:V', 'R2-PM3:V',
                                             'R3-PM1:V', 'R3-PM2:V', 'R3-PM3:V',
                                             'R4-PM1:V', 'R4-PM2:V', 'R4-PM3:V']
    current_phase_magnitudes_header_names = ['R1-PM4:I', 'R1-PM5:I', 'R1-PM6:I',
                                             'R2-PM4:I', 'R2-PM5:I', 'R2-PM6:I',
                                             'R3-PM4:I', 'R3-PM5:I', 'R3-PM6:I',
                                             'R4-PM4:I', 'R4-PM5:I', 'R4-PM6:I']

    # Apparent Impedance measurements for each relay (R1-PA:Z, R2-PA:Z, R3-PA:Z, R4-PA:Z), having values in the 4.8 to 4.9 range
    for header in apparent_impedance_measurements_headers_names:
        df[header+'_in_range(4.8-4.9)'] = np.where((df[header] >= 4.8) & (df[header] <= 4.9), 1, 0)

    # Voltage Phase Angles (PA1:VH – PA3:VH) in the 3.0 range
    for header in voltage_phase_angles_headers_names:
        df[header + '_in_range(3.0)'] = np.where(abs(df[header]-3.0) <= 0.5, 1, 0)

    # Current Phase Angles (PA4:IH – PA6:IH) in the 3.0 range
    for header in current_phase_angles_headers_names:
        df[header + '_in_range(3.0)'] = np.where(abs(df[header]-3.0) <= 0.5, 1, 0)

    # Voltage Phase Magnitudes (PM1:V – PM3:V) in the 3.0 range
    for header in voltage_phase_magnitudes_headers_names:
        df[header + '_in_range(3.0)'] = np.where(abs(df[header]-3.0) <= 0.5, 1, 0)

    # Current Phase Magnitudes (PM4:I – PM6:I) in the 3.0 range
    for header in current_phase_magnitudes_header_names:
        df[header + '_in_range(3.0)'] = np.where(abs(df[header]-3.0) <= 0.5, 1, 0)
    

    return df

def remove_irrelevant_features(df):
    # Remove all NAN columns or replace with desired string
    # This loop iterates over all of the column names which are all NaN
    for column in df.columns[df.isna().any()].tolist():
        # df.drop(column, axis=1, inplace=True)
        df[column] = df[column].fillna(0.0)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.reset_index()
    return df






def model(test_df):
    # Preprocess the test data
    test_df = preprocess(test_df)
    # Remove irrelevant features
    test_df = remove_irrelevant_features(test_df)
    # Vectorize the test data
    test_df = vectorize_df(test_df)
    # Remove irrelevant features
    features_list = test_df.columns.to_list()
    if 'marker' in features_list:
        features_list.remove('marker')
    if 'index' in features_list:
        features_list.remove('index')
    X = test_df[features_list]
    prediction1 = []
    router = pickle.load(open('../P/router.pkl', 'rb'))
    data_index=router.predict(X).astype(int)-1
    for i in range(0,14):
        features = rfecv_binary[i]
        # Filter the test data
        X1 = X[features]
        model = pickle.load(open('../P/sc' + str(i) + '.pkl', 'rb'))
        # Predict the test data and change it to bipolar
        prediction1.append(model.predict(X1) * 2 - 1)
    prediction = []
    # Predict by calculating the weighted mean of the predictions so the model data_index is 100%
    prediction1=np.array(prediction1)
    for i in range(0, len(data_index)-1):
        predictionI= prediction1[:,i]
        predictionI[data_index[i]] = predictionI[data_index[i]] 
        for j in range(0,14):
            if j != data_index[i]:
                predictionI[j] = predictionI[j]*0
        # Calculate the weighted mean
        prediction.append(1 if np.sum(predictionI) > 0 else 0)
    return prediction


def calculate(test_df):

    # Load the test data
    # test_df = pd.read_csv(filename1)
    # Predict the test data
    prediction = model(test_df)
    # results01 = []
    # for value in prediction:
    #     results01.append(scenarios[value])
    
    return prediction

