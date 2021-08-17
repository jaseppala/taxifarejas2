# imports
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import haversine_vectorized, compute_rmse
from memoized_property import memoized_property

import mlflow
from  mlflow.tracking import MlflowClient

import pandas as pd


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.experiment_name = "[DE] [Berlin] [jaseppala] TaxiFareModel + 1.0"

    def set_pipeline(self):
        '''returns a pipelined model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return pipe

    def run(self):
        """set and train the pipeline"""
        pipe = self.set_pipeline()
        self.pipeline = pipe.fit(self.X, self.y)
        

    def evaluate(self, X_test, y_test):
        '''returns the value of the RMSE'''
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse

    MLFLOW_URI = "https://mlflow.lewagon.co/"
    
    @memoized_property
    def mlflow_client(self):
	    mlflow.set_tracking_uri(self.MLFLOW_URI)
	    return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
	    try:
		    return self.mlflow_client.create_experiment(self.experiment_name)
	    except BaseException:
		    return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    def mlflow_create_run(self):
	    self.mlfow_run = self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
	    self.mlflow_client.log_param(self.mlfow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
	    self.mlflow_client.log_metric(self.mlfow_run.info.run_id, key, value)

    def train(self):

        for model in ["linear", "Randomforest"]:
            self.mlflow_create_run()
            self.mlflow_log_metric("rmse", 4.5)
            self.mlflow_log_param("model", model)



if __name__ == "__main__":
    # get data
    from TaxiFareModel.data import get_data, clean_data
    data = get_data()
    # clean data
    df = clean_data(data)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # hold out
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    # train
    trainer = Trainer(X_train, y_train)
    trained_pipe = trainer.run()
    # evaluate
    trainer.evaluate(X_test, y_test)
    
    trainer.train()

    experiment_id = trainer.mlflow_experiment_id

    print(f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}")

