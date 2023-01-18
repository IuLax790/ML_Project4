import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

import warnings
def warn(*args,**kwargs):
    pass
warnings.warn = warn
# Importing the dataset
X = pd.read_csv("C:\\Information_Science\\My_projects\\train.csv")
y=X.pop("is_claim")
print(X)
print(y)

def data_preprocessor(X):
    # Clean max_torque and max_power cols
    X["max_torque_Nm"] = X["max_torque"].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*Nm)").astype('float64')
    X["max_torque_rpm"] = X["max_torque"].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*rpm)").astype('float64')

    X["max_power_bhp"] = X["max_power"].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*bhp)").astype('float64')
    X["max_power_rpm"] = X["max_power"].str.extract(r"([-+]?[0-9]*\.?[0-9]+)(?=\s*rpm)").astype('float64')

    # Drop original cols
    X.drop(["max_torque", "max_power"], axis=1, inplace=True)
    X.drop(["area_cluster","segment", "model","fuel_type"], axis=1, inplace=True)
    X.drop(["engine_type", "is_esc","is_adjustable_steering"], axis=1, inplace=True)
    X.drop(["is_tpms", "is_parking_sensors","is_parking_camera"], axis=1, inplace=True)
    X.drop(["transmission_type", "steering_type","is_front_fog_lights"], axis=1, inplace=True)
    X.drop(["is_rear_window_wiper", "is_rear_window_washer","is_rear_window_defogger"], axis=1, inplace=True)
    X.drop(["is_brake_assist", "is_power_door_locks","is_central_locking"], axis=1, inplace=True)
    X.drop(["is_power_steering", "is_driver_seat_height_adjustable","is_day_night_rear_view_mirror"], axis=1, inplace=True)
    X.drop(["is_ecw", "is_speed_alert","rear_brakes_type"], axis=1, inplace=True)


data_preprocessor(X)



numeric_variables = list(X.dtypes[X.dtypes !="object"].index)
X[numeric_variables].head()
model = RandomForestRegressor(n_estimators=100,oob_score=True,random_state=0)
model.fit(X[numeric_variables],y)


def print_all(X,max_rows=10):
    from IPython.display import display,HTML
    display(HTML(X.to_html(max_rows=max_rows)))
print(print_all(X))



regressor=RandomForestRegressor(n_estimators=3,oob_score=True, random_state=0)
regressor.fit(X,y)
y_oob = regressor.oob_prediction_
print("C-Stat:",roc_auc_score(y,y_oob))
model = regressor.feature_importances_
feature_importances = pd.Series(regressor.feature_importances_,index=X.columns)
feature_importances.plot(kind='barh',figsize=(7,6))
feature_importances.sort_index()
plt.title('Insurance variables by Importance')
print(plt.show())

predict = regressor.predict(X)
result = pd.DataFrame(predict)

X_grid = np.arange(min(X), max(X), 0.1) #we get a vector
X_grid = X_grid.reshape((len(X_grid), 1)) #we need a matrix
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision Tree')
plt.xlabel('Policy Variables')
plt.ylabel('Claiming policy in 6 Months')
print(plt.show())


my_names=[k for k in X.keys()] #get the names of the columns
my_names=my_names[1:-1] #remove the name of the dependent variable (and position)
from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data,filled=True,rounded=True,special_characters=True,feature_names=my_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
z=regressor.predict([[5.8]])
print(z)
