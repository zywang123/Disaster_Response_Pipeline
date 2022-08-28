# Disaster_Response_Pipeline
Author: Zeyao Wang
## Library Used
- panda
- numpy
- re
- pickle
- matplotlib
- seaborn
- sklearn
- operator
- sqlalchemy

## Project Motivation
The project goal is to categorize the disaster message. In this project, I analyzed and built a machine learning pipeline to categorize the real disaster messages so that I could send the messages to an appropriate disaster relief agency. The final web app can input a new message and then get classification results in multiple categories. 

## Files Description
1. ETL Pipeline: in the `process_data.py`
    - load `message` and `categories` datasets
    - merge two datasets
    - clean data
    - store in SQLite databse
2. ML Pipeline: in the `train_classifier.py`
    - load SQLite databse data
    - split dataset into train and test data
    - build a text processing and machine learning pipeline
    - output results on the test set
    - export the model in pickle file
3. Workspace: flask web app
    -  contains visualization of distribution of message genres
    - result category: after enter the message, the category will be highlight by green

## Instructions:
1. Run ETL pipeline:  
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. Run ML pipeline:   
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. Run your web app:  
`python run.py`  

## Acknowledgment
Here are some reference of function use:
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- https://www.projectpro.io/recipes/generate-classification-report-and-confusion-matrix-in-python




