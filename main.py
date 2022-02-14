import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from urllib.parse import urlparse

from scipy import stats
import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred,):
    mse = np.sqrt(mean_squared_error(actual, pred))
    r2 = r2_score(actual, pred)
    return mse, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    df = pd.read_excel('AimoScore_WeakLink_big_scores.xls')

    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    dff = df[filtered_entries]

    df.drop(['EstimatedScore'], axis=1, inplace=True)

    a = df['No_4_Angle_Deviation'] * df['No_6_Angle_Deviation']
    b = df['No_5_Angle_Deviation'] * df['No_7_Angle_Deviation']
    c = df['No_8_Angle_Deviation'] * df['No_11_Angle_Deviation']
    d = df['No_9_Angle_Deviation'] * df['No_12_Angle_Deviation']
    e = df['No_10_Angle_Deviation'] * df['No_13_Angle_Deviation']
    f = df['No_1_NASM_Deviation'] * df['No_2_NASM_Deviation']
    g = df['No_4_NASM_Deviation'] * df['No_5_NASM_Deviation']
    h = df['No_8_NASM_Deviation'] * df['No_9_NASM_Deviation']
    i = df['No_11_NASM_Deviation'] * df['No_12_NASM_Deviation']
    j = df['No_13_NASM_Deviation'] * df['No_14_NASM_Deviation']
    k = df['No_15_NASM_Deviation'] * df['No_16_NASM_Deviation']
    l = df['No_18_NASM_Deviation'] * df['No_19_NASM_Deviation']

    df = df.assign(A4xA6=a, A5xA7=b, A8xA11=c, A9xA12=d, A10xA13=e, N1xN2=f, N4xN5=g, N8xN9=h, N11xN12=i, N13xN4=j,
                   N15xN16=k, N18xN19=l)

    X_train, X_test, Y_train, Y_test = train_test_split(df.iloc[:, 1:53], df['AimoScore'], test_size=0.1,
                                                        random_state=41)

    with mlflow.start_run():
        LR = LinearRegression()
        model = LR.fit(X_train, Y_train)

        y_pred = model.predict(X_test)

        (mse, r2) = eval_metrics(Y_test, y_pred)

        print("mse: %.2f" % mean_squared_error(Y_test, y_pred))
        print("r2: %.2f" % r2_score(Y_test, y_pred))

        mlflow.log_metric("mse",mean_squared_error(Y_test, y_pred) )
        mlflow.log_metric("r2", r2_score(Y_test, y_pred))

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="LinModel")
        else:
            mlflow.sklearn.log_model(model, "model")








