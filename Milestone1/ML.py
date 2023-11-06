#!/usr/bin/env python
from EDA import *       # imports everything from EDA.py, including its variables
# import EDA              # executes the script 
print("ML Job Starting...\n")
# train_df.to_csv("MS_1_Scenario_trainCheck.csv", 
#                 index=False, encoding='utf-8')

# For insurance 
columns = ["Passenger Fare", "Ticket Number", 
           "Name", "Cabin", "NumSiblingSpouse", "NumParentChild", 
           "Age"]
# insurance_train_df = train_df.copy()
insurance_test_df = test_df.copy()
# DropColumn(insurance_train_df, columns)
DropColumn(insurance_test_df, columns)
# # Placeholder incase the column creation in the SVM portion starts acting up
# # For now not in use
# if not {'Prediction'}.issubset(insurance_train_df.columns):   # Prevent column duplication
#     insurance_train_df.insert(8,"Prediction", 0, allow_duplicates=bool)
# if not {'Prediction'}.issubset(insurance_test_df.columns):   # Prevent column duplication
#     insurance_test_df.insert(8,"Prediction", 0, allow_duplicates=bool)



# Before modelling can begin, the dataframe needs to be cleaned.
# Remove columns that are not desireable
columns = ["Passenger ID", "Passenger Fare", "Ticket Number", 
           "Name", "Cabin", "NumSiblingSpouse", "NumParentChild", 
           "Age"] #  "FareBand" "Embarkation Country" "AgeBand"
DropColumn(train_df, columns)
DropColumn(test_df, columns)

# Preparing csv files for model creation
# Exporting train_df and test_df Survived column to separate csv files
train_df.to_csv("MS_1_Scenario_train_survived.csv", columns=['Survived'], 
                index=False, encoding='utf-8')
test_df.to_csv("MS_1_Scenario_test_survived.csv", columns=['Survived'], 
               index=False, encoding='utf-8')

# Drop Survived column
columns = ["Survived"]
DropColumn(train_df, columns)
DropColumn(test_df, columns)

# train_survive_df and test_survive_df contains the Survived column originally 
# from train_df and test_df respectively.
train_survive_df = pd.read_csv("MS_1_Scenario_train_survived.csv")
test_survive_df = pd.read_csv("MS_1_Scenario_test_survived.csv")

# print(train_df.head())
# print(test_df.head())
# print(train_survive_df.head())
# print(test_survive_df.head())

## Gaussian Naive Bayes
# Gaussian Naive Bayes assumes that each feature or predictor has an independent
# capacity of predicting the output variable.
gaussian = GaussianNB()
# Trains model based on training set, training Survived column
gaussian.fit(train_df, train_survive_df.values.ravel())   
# Produces 1D column array of predictions
gaussianPred = gaussian.predict(test_df)

## Logistic Regression
# Models the probability of an event taking place
logReg = LogisticRegression(class_weight=None, max_iter=36, 
                            multi_class='multinomial', penalty=None, 
                            solver='newton-cg')
logReg.fit(train_df, train_survive_df.values.ravel())   
logRegPred = logReg.predict(test_df)

## Random Forest
# Combines the output of multiple decision trees to reach a single 
# result (Majority voting).
# # n_estimators are number of trees, more = slower (may produce better results)
rdmForest = RandomForestClassifier(bootstrap=True, max_depth=7, max_features=None, 
                                   min_samples_leaf=8, min_samples_split=8, 
                                   n_estimators=64)       
rdmForest.fit(train_df, train_survive_df.values.ravel()) 
rdmForestPred = rdmForest.predict(test_df)

## k-Nearest Neighbors
# The average the k nearest neighbors are taken to make a prediction about 
# a classification.
# n_neighbors are the number of nearest neighbors to take (more not always better).
knn = KNeighborsClassifier(n_neighbors = 28)        
knn.fit(train_df, train_survive_df.values.ravel())  
knnPred = knn.predict(test_df)

## Support Vector Machines
# Finds the optimal hyperplane in an N-dimensional space that can separate the 
# data points in different classes in the feature space.
# kernel = type of algo, gamma = kernel coef, degree of poly kernel ignored by 
# other kernel; scale  
svm = SVC(kernel='poly', gamma='auto', degree=5, probability=True)   
svm.fit(train_df, train_survive_df.values.ravel())         
svmPred = svm.predict(test_df)



# Insurance
# Wrap the SVC classifier with CalibratedClassifierCV
# calibrated_svc = CalibratedClassifierCV(estimator=svm, method='sigmoid')
# calibrated_svc.fit(train_df, train_survive_df.values.ravel())
# # Get class probabilities
# calibrated_svc_prediction_train = np.round(calibrated_svc.predict_proba(train_df), 3)
# calibrated_svc_prediction_test = np.round(calibrated_svc.predict_proba(test_df), 3)

# trainPred = svm.predict(train_df)
# insurance_train_df['Prediction'] = trainPred.tolist()
# insurance_test_df['Prediction'] = svmPred.tolist()
# insurance_train_df['Death Probability'] = calibrated_svc_prediction_train[:, 0].tolist()
# insurance_train_df['Alive Probability'] = calibrated_svc_prediction_train[:, 1].tolist()
# insurance_test_df['Death Probability'] = calibrated_svc_prediction_test[:, 0].tolist()
# insurance_test_df['Alive Probability'] = calibrated_svc_prediction_test[:, 1].tolist()
# insurance_train_df.to_csv('insurance_train_predictions.csv', index=False)
# insurance_test_df.to_csv('insurance_test_predictions.csv', index=False)


## Decision Tree
# Flowchart-like tree structure, node represents a feature(or attribute), 
# branches represents a decision rule, and each leaf node represents the outcome.
# Max depth of the tree, more caused overfitting, less causes underfitting
decTree = DecisionTreeClassifier(max_depth=5)            
trees = decTree.fit(train_df, train_survive_df.values.ravel())
decTreePred = decTree.predict(test_df)      

# from sklearn.tree import export_graphviz
# from six import StringIO  
# from IPython.display import Image  
# import pydotplus
# feature_cols = ['Ticket Class',	'Embarkation Country', 'Title',	'AgeGroup',	'Gender',	'NumRelativeBrought']
# dot_data = StringIO()
# export_graphviz(trees, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True,feature_names = feature_cols,class_names=['Dead','Alive'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# graph.write_png('Decision_Tree.png')
# Image(graph.create_png())

# Insurance            
# trainPred = decTree.predict(train_df)
# insurance_train_decTreePred = np.round(decTree.predict_proba(train_df),3)
insurance_test_decTreePred = np.round(decTree.predict_proba(test_df),3)
# insurance_train_df['Prediction'] = trainPred.tolist()
insurance_test_df['Prediction'] = decTreePred.tolist()
# insurance_train_df['Death Probability'] = insurance_train_decTreePred[:, 0].tolist()
# insurance_train_df['Alive Probability'] = insurance_train_decTreePred[:, 1].tolist()
# insurance_test_df['Death Probability'] = insurance_test_decTreePred[:, 0].tolist()
insurance_test_df['Alive Probability'] = insurance_test_decTreePred[:, 1].tolist()
# insurance_train_df.to_csv('insurance_train_predictions.csv', index=False)
insurance_test_df.to_csv('insurance_test_predictions.csv', index=False)



print("ML Job Done")
# Prevents auto closure of command promt for exe
# k=input("Input a character followed by pressing Enter to exit") 