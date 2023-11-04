#!/usr/bin/env python
# imports everything from ML.py, including its variables
# from ML import *
# executes the scripts inside the files
# import EDA
from HeaderFile import *
# import EDA
from EDA import train_df_B4mod, test_df_B4mod, ageMean
from EDA import train_df_B4drop, test_df_B4drop, titles 
import ML
from ML import train_df, test_df, test_survive_df
from ML import gaussian, gaussianPred, logReg, logRegPred
from ML import rdmForest, rdmForestPred, knn, knnPred
from ML import svm, svmPred, decTree, decTreePred
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
# train_df_B4mod   original dataframe set from csv
# test_df_B4mod
# train_df_B4drop  dataframe after modification, before dropping columns
# test_df_B4drop


## EDA
def EDA_corr_b4_ML():
  # Displays correlation graph
  corr(train_df_B4drop)
  plt.show()

def EDA_title_age():
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
  plt.show()

def EDA_TC_age_after_fill():
  # Ticket Class and Age according w.r.t to Survived
  grid = sns.FacetGrid(train_df_B4drop, col='Survived', row='Ticket Class', 
                      height=2.2, aspect=1.6)
  grid.map(plt.hist, 'Age', bins=20)
  grid.add_legend()
  plt.show()

def EDA_TC_age_b4_fill():
  grid = sns.FacetGrid(train_df_B4mod, col='Survived', row='Ticket Class', 
                      height=2.2, aspect=1.6)
  grid.map(plt.hist, 'Age', bins=20)
  grid.add_legend()
  plt.show()

def EDA_age_after_fill():
  # Age plot after filling in entries
  grid = sns.FacetGrid(train_df_B4drop, col='Survived', height=2.2, aspect=1.6)
  grid.map(plt.hist, 'Age', bins=20)
  grid.add_legend()
  plt.show()

def EDA_age_b4_fill():
  # Age plot before filling in entries
  grid = sns.FacetGrid(train_df_B4mod, col='Survived', height=2.2, aspect=1.6)
  grid.map(plt.hist, 'Age', bins=20)
  grid.add_legend()
  plt.show()

def EDA_TC_prob():
  # Ticket Class prob
  ProbabilityCheck(train_df_B4mod, 'Ticket Class')
  plt.show()

def EDA_cabin_prob():
  # Cabin prob
  ProbabilityCheck(train_df_B4mod, 'Cabin')
  plt.show()

def EDA_gender_prob():
  # Gender prob
  ProbabilityCheck(train_df_B4mod, 'Gender')
  plt.show()

def EDA_SS_prob():
  # NumSiblingSpouse prob
  ProbabilityCheck(train_df_B4mod, 'NumSiblingSpouse')
  plt.show()

def EDA_PC_prob():
  # NumParentChild prob
  ProbabilityCheck(train_df_B4mod, 'NumParentChild')
  plt.show()

def EDA_EC_prob():
  # Embarkation Country prob
  ProbabilityCheck(train_df_B4mod, 'Embarkation Country')
  plt.show()

def EDA_passenger_fare():
  # Passenger Fare plot
  grid = sns.FacetGrid(train_df_B4mod, col='Survived', height=2.2, aspect=1.6)
  grid.map(plt.hist, 'Passenger Fare', bins=[0,100,200,300,400,500,600])
  grid.add_legend()
  plt.show()

def EDA_corr_ori():
  # Displays correlation graph
  corr(train_df_B4mod)
  plt.show()

def EDA_description():
  ## Using original training data set
  # print table for statistical analysis
  train_df_B4mod_describe = train_df_B4mod.describe(include='all')
  test_df_B4mod_describe = test_df_B4mod.describe(include='all')
  print("EDA_Training_Set_Description\n", train_df_B4mod_describe)
  print("EDA_Test_Set_Description\n", test_df_B4mod_describe)



## ML
def ML_corr():
  # Displays correlation graph
  corr(train_df)
  plt.show()

def ML_gaussNB():
  # Gaussian Naive Bayes     
  PerformanceMetric(gaussianPred, 'Gaussian Naive Bayes', test_survive_df) 
  plt.show()

def ML_logReg():
  # Logistic Regression    
  PerformanceMetric(logRegPred, 'Logistic Regression', test_survive_df)  
  plt.show()

def ML_RanForest():
  # Random Forest         
  PerformanceMetric(rdmForestPred, 'Random Forest', test_survive_df)       
  plt.show()

def ML_KNN():
  # K-Nearest Neighbors
  PerformanceMetric(knnPred, 'k-Nearest Neighbors', test_survive_df)  
  plt.show()

def ML_SVM():
  # Support Vector Machine
  PerformanceMetric(svmPred, 'Support Vector Machines', test_survive_df)  
  plt.show()

def ML_DecTree():
  # Decision Tree
  PerformanceMetric(decTreePred, 'Decision Tree', test_survive_df)      
  plt.show()

def ML_ranking(preci: bool):
  
  if preci:
    acc_gaussian = round(precision_score(test_survive_df, gaussianPred) * 100, 2) 
    acc_logReg = round(precision_score(test_survive_df, logRegPred) * 100, 2)  
    acc_rdmForest = round(precision_score(test_survive_df, rdmForestPred) * 100, 2) 
    acc_knn = round(precision_score(test_survive_df, knnPred) * 100, 2)
    acc_svm = round(precision_score(test_survive_df, svmPred) * 100, 2) 
    acc_decTree = round(precision_score(test_survive_df, decTreePred) * 100, 2)
  else:
    acc_gaussian = round(gaussian.score(test_df, test_survive_df) * 100, 2) 
    acc_logReg = round(logReg.score(test_df, test_survive_df) * 100, 2)  
    acc_rdmForest = round(rdmForest.score(test_df, test_survive_df) * 100, 2) 
    acc_knn = round(knn.score(test_df, test_survive_df) * 100, 2)
    acc_svm = round(svm.score(test_df, test_survive_df) * 100, 2) 
    acc_decTree = round(decTree.score(test_df, test_survive_df) * 100, 2)

  ranktype = "Precision" if preci else "Accuracy"
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
  plt.title('Model {} (Ascending Order)'.format(ranktype))
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
    print("\nVisuals Menu:")
    print("1. Display visuals for EDA")
    print("2. Display visuals for ML")
    print("3. Exit Menu")
    option = input("Select an option (1/3): ")

    if option == '1':
      while True:
        print("\nEDA Visuals Menu:")
        print("1. Display description of entries")
        print("2. Display correlation heatmap of original entries")
        print("3. Display Passenger Fare probability")
        print("4. Display Embarkation Country probability")
        print("5. Display NumParentChild probability")
        print("6. Display NumSiblingSpouse probability")
        print("7. Display Gender probability")
        print("8. Display Cabin probability")
        print("9. Display Ticket Class probability")
        print("10. Display Age Probability (original entries)")
        print("11. Display Age Probability (after filling '0' age entries)")
        print("12. Display Ticket Class and Age probability (original entries)")
        print("13. Display Ticket Class and Age probability (after filling '0' age entries)")
        print("14. Display Titles and mean age")
        print("15. Display correlation heatmap before ML")
        print("16. Exit Menu")
        option = input("Select an option (1/16): ")

        if option == '1':
          EDA_description()
        elif option == '2':
          EDA_corr_ori()
        elif option == '3':
          EDA_passenger_fare()
        elif option == '4':
          EDA_EC_prob()
        elif option == '5':
          EDA_PC_prob()
        elif option == '6':
          EDA_SS_prob()
        elif option == '7':
          EDA_gender_prob()
        elif option == '8':
          EDA_cabin_prob()
        elif option == '9':
          EDA_TC_prob()
        elif option == '10':
          EDA_age_b4_fill()
        elif option == '11':
          EDA_age_after_fill()
        elif option == '12':
          EDA_TC_age_b4_fill()
        elif option == '13':
          EDA_TC_age_after_fill()
        elif option == '14':
          EDA_title_age()
        elif option == '15':
          EDA_corr_b4_ML()
        elif option == '16':
          break
        else:
          print("Invalid option. Please select a valid option (1-16).\n")


    elif option == '2':
      while True:
        print("\nML Visuals Menu:")
        print("1. Display correlation heatmap of features for ML")
        print("2. Display Gaussian Naive Bayes results ")
        print("3. Display Logistic Regression results")
        print("4. Display Random Forest results")
        print("5. Display K-Nearest Neighbors results")
        print("6. Display Support Vector Machine results")
        print("7. Display Decision Tree results")
        print("8. Display model ranking based on precision")
        print("9. Display model ranking based on accuracy")
        print("10. Exit Menu")
        option = input("Select an option (1/10): ")

        if option == '1':
          ML_corr()
        elif option == '2':
          ML_gaussNB()
        elif option == '3':
          ML_logReg()
        elif option == '4':
          ML_RanForest()
        elif option == '5':
          ML_KNN()
        elif option == '6':
          ML_SVM()
        elif option == '7':
          ML_DecTree()
        elif option == '8':
          ML_ranking(1)
        elif option == '9':
          ML_ranking(0)
        elif option == '10':
          break
        else:
          print("Invalid option. Please select a valid option (1-10).\n")

    elif option == '3':
      break
    else:
      print("Invalid option selected...\n")
    
if __name__ == "__main__":
  main()