This project focuses on predicting the tags of algorithmic exercises from the Codeforces platform using machine learning and natural language processing (NLP) techniques. 
The dataset with all problem descriptions, metadata and validated participant solutions can be found here https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval.
For this project, we only use a subset containing 4982 unique problems, each provided in JSON format.

Project Overview :
The goal is to develop a model that predicts the relevant tags for a given problem. Each exercise is associated with one or multiple tags, such as:
- math
- graphs
- strings
- number theory
- trees
- geometry
- games
- probabilities
  
For this challenge, we only focus on the 8 tags listed above.

How to use xgboost_tags_predictor.py : python <xgboost_tags_predictor.py> <input.json|input_folder> <output.csv>


