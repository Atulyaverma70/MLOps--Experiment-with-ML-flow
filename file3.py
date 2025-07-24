import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import dagshub

dagshub.init(repo_owner='Atulyaverma70', repo_name='MLOps--Experiment-with-ML-flow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Atulyaverma70/MLOps--Experiment-with-ML-flow.mlflow")

wine=load_wine()
X=wine.data
y=wine.target

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.10,random_state=42)
max_depth=15
n_estimators=8

mlflow.autolog()
mlflow.set_experiment('MLOps-Exp2')
with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators, random_state=42)
    rf.fit(X_train,y_train)

    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)


    # creating confusion matrix plot
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    #log artifacts using mlflow
    mlflow.log_artifact(__file__)

    mlflow.set_tags({"Author": 'Atulya', "Project":'Wine Classification'})




    print(accuracy)