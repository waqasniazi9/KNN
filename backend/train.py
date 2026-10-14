from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
FEATURES = NUMERIC + CATEGORICAL
ROOT = Path(__file__).parent

def pipeline():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocess = ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
    return Pipeline([("preprocess", preprocess), ("scale", StandardScaler()), ("knn", KNeighborsClassifier())])

def train():
    data = pd.read_csv(ROOT / "data/train.csv")
    y = data.Loan_Status.map({"N": 0, "Y": 1})
    if y.isna().any():
        raise ValueError("Unexpected or missing target labels.")
    X_train, X_test, y_train, y_test = train_test_split(data[FEATURES], y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    experiments=[]
    for k in [3,5,7]:
        scores=cross_val_score(pipeline().set_params(knn__n_neighbors=k), X_train, y_train, cv=cv, scoring="accuracy")
        experiments.append({"k":k,"CV_accuracy":float(scores.mean()),"CV_SD":float(scores.std())})
    default=pipeline().fit(X_train,y_train)
    search=GridSearchCV(pipeline(), {"knn__n_neighbors":[3,5,7,9,11,15,21]}, cv=cv, scoring="accuracy", n_jobs=1)
    search.fit(X_train,y_train)
    tuned=search.best_estimator_
    predictions={"default":default.predict(X_test),"tuned":tuned.predict(X_test)}
    report={"rows":len(data),"train_rows":len(X_train),"test_rows":len(X_test),"selected_k":int(search.best_params_["knn__n_neighbors"]),"best_CV_accuracy":float(search.best_score_)}
    for name,p in predictions.items():
        report[name]={"accuracy":float(accuracy_score(y_test,p)),"confusion_matrix":confusion_matrix(y_test,p,labels=[0,1]).tolist()}
    report["majority_baseline_accuracy"]=float(accuracy_score(y_test,np.repeat(y_train.mode().iloc[0],len(y_test))))
    comparison=X_test.copy()
    comparison["Actual"]=y_test
    for name,p in predictions.items(): comparison[name]=p
    return default,tuned,report,pd.DataFrame(experiments),comparison,data,search

def save():
    import joblib
    default,tuned,report,experiments,comparison,data,search=train()
    output=ROOT/"artifacts"; output.mkdir(exist_ok=True)
    joblib.dump(tuned,output/"knn_pipeline.joblib")
    (output/"metrics.json").write_text(json.dumps(report,indent=2))
    experiments.to_csv(output/"k_experiments.csv",index=False)
    comparison.to_csv(output/"test_predictions.csv",index=False)
    pd.DataFrame(search.cv_results_).to_csv(output/"grid_search.csv",index=False)
    return report
if __name__=="__main__": print(json.dumps(save(),indent=2))
