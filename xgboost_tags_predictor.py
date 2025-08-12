import os
import sys
import csv
import json
import joblib
import numpy as np
from scipy.sparse import hstack

# This function loads tags and model constants
def load_constants(path="constants.json"):
    with open(path, "r") as f:
        constants=json.load(f)
    return constants

# This function loads vectorizers, scaler and models
def load_models(tags):
    tfidf_desc=joblib.load(f"saved_models/tfidf_desc.joblib")
    tfidf_code=joblib.load(f"saved_models/tfidf_code.joblib")
    scaler=joblib.load(f"saved_models/scaler.joblib")
    models={}
    for tag in tags:
        models[tag]=joblib.load(f"saved_models/xgb_model_{tag}.joblib")
    return tfidf_desc, tfidf_code, scaler, models

# This function preprocess the json input and returns a vector 
def preprocess_input(data, tfidf_desc, tfidf_code, scaler, mean_difficulty):
    # Checking if prob_desc_description is present
    if "prob_desc_description" not in data or not data["prob_desc_description"]:
        raise ValueError(f"Missing prob_desc_description in input JSON") 
    
    # Checking if prob_desc_description is present
    if "source_code" not in data or not data["source_code"]:
        raise ValueError(f"Missing source_code in input JSON")
         
    # Checking if prob_desc_description is present
    if "difficulty" not in data or not data["difficulty"] or data["difficulty"]==-1:
        data["difficulty"]=mean_difficulty

    # Applying vectorizers and scaler to the data
    X_desc=tfidf_desc.transform([data["prob_desc_description"]])
    X_code=tfidf_code.transform([data["source_code"]])
    X_difficulty=scaler.transform(np.array([[data["difficulty"]]]))

    # Concatenation of the vectors
    X=hstack([X_desc, X_code, X_difficulty]).toarray()

    return X

# This function uses the models to do a prediction, and save the predicted tags in a list
def predict(X_final, models, prediction_threshold):
    predicted_tags=[]
    for tag, model in models.items():
        proba=model.predict_proba(X_final)[:, 1][0]
        if proba>=prediction_threshold:
            predicted_tags.append(tag)
    return predicted_tags


def main():
    if len(sys.argv)!=3:
        print(f"Usage: python <xgboost_tags_predictor.py> <input.json|input_folder> <output.csv>")
        sys.exit(1)
    input_path=sys.argv[1]
    output_path=sys.argv[2]

    try:
        constants=load_constants()
        tags=constants["tags"]
        prediction_threshold=constants["prediction_threshold"] 
        mean_difficulty=constants["mean_difficulty"]
        tfidf_desc, tfidf_code, scaler, models=load_models(tags)

        # Checking whether the input is a .json file or a folder containing .json files
        if os.path.isfile(input_path) and input_path.endswith(".json"):
            with open(input_path, "r") as f:
                data=json.load(f)
            X=preprocess_input(data, tfidf_desc, tfidf_code, scaler,mean_difficulty)
            predicted_tags=predict(X, models, prediction_threshold)
            with open(output_path, "w", newline='') as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(["filename", "predicted_tags"])
                writer.writerow([os.path.basename(input_path), ";".join(predicted_tags)])

        # If the input path is a folder, we process all of them sequentially and save all the predictions in a csv file  
        elif os.path.isdir(input_path):
            json_files = [f for f in os.listdir(input_path) if f.endswith(".json")]
            with open(output_path, "w", newline='') as csvfile:
                writer=csv.writer(csvfile)
                writer.writerow(["filename", "predicted_tags"])
                for filename in json_files:
                    filepath=os.path.join(input_path, filename)
                    try:
                        with open(filepath, "r") as f:
                            data=json.load(f)
                        X=preprocess_input(data, tfidf_desc, tfidf_code, scaler,mean_difficulty)
                        predicted_tags=predict(X, models, prediction_threshold)
                        writer.writerow([filename, ";".join(predicted_tags)])
                    except Exception as e:
                        print(f"[ERROR] Failed to process {filename}: {e}")

        else:
            print(f"[ERROR] Input path is not a JSON file or a directory")
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__=="__main__":
    main()