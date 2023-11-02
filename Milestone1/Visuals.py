#!/usr/bin/env python
# imports everything from ML.py, including its variables
# from ML import *
# executes the scripts inside the files
# import EDA
from HeaderFile import *

# train_df_B4mod   original dataframe set from csv
# test_df_B4mod
# train_df_B4drop  dataframe after modification, before dropping columns
# test_df_B4drop

def EDA_Visuals():
  # import EDA
  from EDA import train_df_B4mod, test_df_B4mod, ageMean
  from EDA import train_df_B4drop, test_df_B4drop, titles 
  print('-'*20, "Results from EDA", '-'*20, "\n")
  ## Using original training data set
  # print table for statistical analysis
  train_df_B4mod_describe = train_df_B4mod.describe(include='all')
  test_df_B4mod_describe = test_df_B4mod.describe(include='all')
  NonTruncDisplay(train_df_B4mod_describe, "EDA_Training_Set_Description")
  NonTruncDisplay(test_df_B4mod_describe, "EDA_Test_Set_Description")

  print("Displaying EDA Figures...")
  # Displays correlation graph
  corr(train_df_B4drop)

  # Average Age of titles
  fig, ax = plt.subplots(figsize=(6, 3))
  totalIndex = 0
  yplacing = 0.9
  ax.text(0.45, yplacing, 
          "----Average valid ages of titles in the training set----", 
          fontsize=10, ha="center")
  for index in range(len(titles)):
    age_count = "{} counts of".format(ageMean[index][1])
    title = "{}:".format(titles[index])
    age_value = "{}".format(ageMean[index][2])

    # Combine the three strings into one
    result = " ".join([age_count, title, age_value])

    yplacing-=(0.1)
    ax.text(0.1, yplacing, result, fontsize=10, ha="left")
    totalIndex += ageMean[index][1]

  # For those that doesn't fall in the 'titles' list
  age_count = "{} counts of".format(ageMean[len(titles)][1])
  title = "passengers without titles"
  age_value = "{}".format(ageMean[len(titles)][2])

  # Combine the three strings into one
  result = " ".join([age_count, title, age_value])
  ax.text(0.1, yplacing-0.1, result, fontsize=10, ha="left")
  totalIndex += ageMean[len(titles)][1]
  ax.text(0.1, yplacing-0.2, "Entries with '0' age: 167", 
          fontsize=10, ha="left")
  ax.text(0.1, yplacing-0.3, "Total entries: {}".format(totalIndex+167), 
          fontsize=10, ha="left")
  # Remove axis ticks and labels
  ax.axis("off")

  # Ticket Class and Age according w.r.t to Survived
  grid = sns.FacetGrid(train_df_B4drop, col='Survived', row='Ticket Class', 
                      height=2.2, aspect=1.6)
  grid.map(plt.hist, 'Age', bins=20)
  grid.add_legend()
  grid = sns.FacetGrid(train_df_B4mod, col='Survived', row='Ticket Class', 
                      height=2.2, aspect=1.6)
  grid.map(plt.hist, 'Age', bins=20)
  grid.add_legend()

  # Age plot after filling in entries
  grid = sns.FacetGrid(train_df_B4drop, col='Survived', height=2.2, aspect=1.6)
  grid.map(plt.hist, 'Age', bins=20)
  grid.add_legend()
  # Age plot before filling in entries
  grid = sns.FacetGrid(train_df_B4mod, col='Survived', height=2.2, aspect=1.6)
  grid.map(plt.hist, 'Age', bins=20)
  grid.add_legend()


  # Ticket Class prob
  ProbabilityCheck(train_df_B4mod, 'Ticket Class')
  # Cabin prob
  ProbabilityCheck(train_df_B4mod, 'Cabin')
  # Gender prob
  ProbabilityCheck(train_df_B4mod, 'Gender')
  # NumSiblingSpouse prob
  ProbabilityCheck(train_df_B4mod, 'NumSiblingSpouse')
  # NumParentChild prob
  ProbabilityCheck(train_df_B4mod, 'NumParentChild')
  # Embarkation Country prob
  ProbabilityCheck(train_df_B4mod, 'Embarkation Country')
  # Passenger Fare plot
  grid = sns.FacetGrid(train_df_B4mod, col='Survived', height=2.2, aspect=1.6)
  grid.map(plt.hist, 'Passenger Fare', bins=[0,100,200,300,400,500,600])
  grid.add_legend()
  # Displays correlation graph
  corr(train_df_B4mod)

  print("Close all Figures to return to selection screen\n")
  # Display the figure
  plt.show()


def ML_Visuals():
  # import ML
  from ML import train_df, test_df, test_survive_df
  from ML import gaussian, gaussianPred, logReg, logRegPred
  from ML import rdmForest, rdmForestPred, knn, knnPred
  from ML import svm, svmPred, decTree, decTreePred

  print("Displaying ML Figures...")
  ## ML visualisation
  # Representing data from ML 
  print('-'*20, "Results from ML", '-'*20, "\n")
  # Displays correlation graph
  corr(train_df)

  # Gaussian Naive Bayes     
  PerformanceMetric(gaussianPred, 'Gaussian Naive Bayes', test_survive_df)   
  acc_gaussian = round(gaussian.score(test_df, test_survive_df) * 100, 2)

  # Logistic Regression    
  PerformanceMetric(logRegPred, 'Logistic Regression', test_survive_df)   
  acc_logReg = round(logReg.score(test_df, test_survive_df) * 100, 2)

  # Random Forest         
  PerformanceMetric(rdmForestPred, 'Random Forest', test_survive_df)         
  acc_rdmForest = round(rdmForest.score(test_df, test_survive_df) * 100, 2)

  # K-Nearest Neighbors
  PerformanceMetric(knnPred, 'k-Nearest Neighbors', test_survive_df)   
  acc_knn = round(knn.score(test_df, test_survive_df) * 100, 2)

  # Support Vector Machine
  PerformanceMetric(svmPred, 'Support Vector Machines', test_survive_df)  
  acc_svm = round(svm.score(test_df, test_survive_df) * 100, 2)

  # Decision Tree
  PerformanceMetric(decTreePred, 'Decision Tree', test_survive_df)       
  acc_decTree = round(decTree.score(test_df, test_survive_df) * 100, 2)

  # Ranking models
  models = pd.DataFrame({
      'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
                'Random Forest', 'Gaussian Naive Bayes', 'Decision Tree'],
      'Scores': [acc_svm, acc_knn, acc_logReg,
                acc_rdmForest, acc_gaussian, acc_decTree]})

  models = models.sort_values("Scores", ascending=True)

  # Create a bar plot
  plt.figure(figsize=(12, 4))
  plt.bar(models['Model'], models['Scores'], color='skyblue')
  plt.xlabel('Score')
  plt.title('Model Accuracy (Ascending Order)')
  # Display the scores on top of the bars
  for i, score in enumerate(models['Scores']):
      plt.text(i, score/2, f'{score:.2f}', ha='center', va='bottom', fontsize=17, color="#FF0000")

  print("Close all Figures to return to selection screen\n")
  # Display all figures
  plt.show()

  # Prevents auto closure of command promt for exe
  # k=input("Input a character followed by pressing Enter to exit") 


def main():
  while True:
    print("Visuals Menu:")
    print("1. Display visuals for EDA")
    print("2. Display visuals for ML")
    print("3. Exit Menu")
    option = input("Select an option (1/3): ")

    if option == '1':
      EDA_Visuals()
    elif option == '2':
      ML_Visuals()
    elif option == '3':
      break
    else:
      print("Invalid option selected...\n")
    
if __name__ == "__main__":
  main()