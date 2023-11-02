#!/usr/bin/env python3

## Libraries
# Additional 
import os
import sys

# These libraries are required to perform dataset manupulation.
import pandas as pd
import numpy as np
# Libraries for data visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Libraries for performance check
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from math import sqrt

# Libraries for Machine Learning
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from scipy.stats import randint


###### EDA Functions
# Removes '$' and rounds the value to 2 d.p. for Passenger Fare column
# df_set refers to the dataframe (train_df, test_df)
def RemoveDollarSign(df_set: pd.DataFrame):
  # Cast to string first
  df_set["Passenger Fare"] = df_set["Passenger Fare"].astype(str)
  # Replace the dollar sign with nothing, then cast to appropriate data type
  df_set['Passenger Fare'] = df_set['Passenger Fare'].str.replace('$','').astype(float)
  # Round the fare to 2 decimal places
  df_set['Passenger Fare'] = df_set['Passenger Fare'].round(2)


# Remove numeric from alphanumeric values in Cabin
# df_set refers to train_df or test_df
# Takes in the dataframe and replaces each alphanumeric entry
# with the first alphabet
def CabinFilter(df_set: pd.DataFrame):
  for index, row in df_set.iterrows():
      cabinStr = row['Cabin']
      if cabinStr[0] != "0": # Does not concat for Cabin 0
        df_set.at[index, 'Cabin'] = cabinStr[0]

# Function to convert categorical non-numerical binary options 
# (male, female; Yes, No) to categorical numerical binary options
# df_set refers to train_df or test_df
# column refers to the characteristic headers, e.g., 'Survived'
# opt1 and opt1 refers to the categorical non-numerical binary options
def Conversion(df_set: pd.DataFrame, column: object, opt1: object, opt2: object):
  for index, row in df_set.iterrows():
    colStr = row[column]
    if colStr == opt1:                  # "male", "Yes"
      df_set.at[index, column] = "1"
    elif colStr == opt2:                # "female", "No"
      df_set.at[index, column] = "0"


# Converts dataframe columns into type int
def ToTypeInt(df_set: pd.DataFrame, column: object):
    df_set[column] = df_set[column].astype(np.int64)


# Removes Embarkation Country entries that aren't 'A' to 'Z' or 'a' to 'z' 
# and categorizes the rest. Known countries: C, Q, S,  to 1, 2, 3. 
# Everything else is group 4
# df_set refers to the dataframe (train_df, test_df)
def RemoveOutlier(df_set: pd.DataFrame):
  for index, row in df_set.iterrows():
    #if (row['Embarkation Country'].find('0') != -1):
    #  df_set = df_set.drop(index) # remove row
    if (ord(row['Embarkation Country'].upper()) > 64 
        and ord(row['Embarkation Country'].upper()) < 91):
        # Within A to Z Countries, assumes a to z as valid as well
      if (row['Embarkation Country'] == "C" or 
          row['Embarkation Country'] == "c"):
        df_set.at[index, 'Embarkation Country'] = 1
      elif (row['Embarkation Country'] == "Q" or 
            row['Embarkation Country'] == "q"):
        df_set.at[index, 'Embarkation Country'] = 2
      elif (row['Embarkation Country'] == "S" or 
            row['Embarkation Country'] == "s"):
        df_set.at[index, 'Embarkation Country'] = 3
      else:
        df_set.at[index, 'Embarkation Country'] = 4     # outside of known countries
    # Not an alphabet
    else:
      df_set = df_set.drop(index) # remove row
  ToTypeInt(df_set, 'Embarkation Country')
  return df_set               # for some reason it wasn't updating the orginal dataset


# Amends entries for the Title column and finds the age for each title from train_df.
# Inserts average age of corresponding titles for entries with "0" in the Age column.
# Groups ages according to certian intervals.
# df_set refers to train_df or test_df
# trainingSet refers to bool expression; trainingSet is True if df_set is train_df
# titles are categorical numeric expressions following the index of the titles list
# ageMean contains the sum of ages, number of people of that title, average age
def Amend_Title_Age(df_set: pd.DataFrame, trainingSet: bool, titles, ageMean):
  # Prevents multiple creation
  if not {'Title', 'AgeGroup'}.issubset(df_set.columns):
    # add column Title after name
    df_set.insert(7, "Title", 0, allow_duplicates=True)
    # add column AgeGroup
    df_set.insert(9, "AgeGroup", 0, allow_duplicates=True)

  # iterate thru rows
  for index, row in df_set.iterrows():
    # iterate thru the titles -> see if it matches the name
    for t in titles:
      if (row['Name'].find(t) != -1):
        df_set.at[index,'Title'] = int(titles.index(t))
        if row['Age'] != 0 and trainingSet == True:     # find age from train_df only
          ageMean[titles.index(t)][1] += 1              # + 1 count of title with valid age
          ageMean[titles.index(t)][0] += row['Age']     # sum of all ages of a particular title
        break
      elif titles.index(t) == (int(len(titles))-1):
        df_set.at[index,'Title']= int(len(titles))
        if row['Age'] != 0 and trainingSet == True:     # find age from train_df only
          ageMean[len(titles)][1] += 1                  # + 1 count of title with valid age
          ageMean[len(titles)][0] += row['Age']         # sum of all ages of a particular title

 
  # For those with titles in the 'titles' list; Only for train_df; Only for those with age > 0
  if trainingSet == True:                           
    totalIndex = 0
    for index in range(len(titles)):
      ageMean[index][2] = int(np.ceil(ageMean[index][0]/ ageMean[index][1]))
      # print("Average age of", ageMean[index][1], "names with", titles[index], 
      #                           "are", ageMean[index][2])
      totalIndex += ageMean[index][1]

    # For those that doesn't fall in the 'titles' list
    ageMean[len(titles)][2] = int(np.ceil(ageMean[len(titles)][0]/ ageMean[len(titles)][1]))
    # print("Average age of the other", ageMean[len(titles)][1], 
    #           "names are", ageMean[len(titles)][2])
    totalIndex += ageMean[len(titles)][1]
    # print(totalIndex + 167, "entries")

  for index, row in df_set.iterrows():
    for i in range(len(titles)):
      if row['Title'] == i and row['Age'] == 0:        # title and "0" age detected
        df_set.at[index, 'Age'] = ageMean[i][2]        # average age of passengers with same title

    # For those that doesn't fall in the 'titles' list
    if row['Title'] == len(titles) and row['Age'] == 0:
        df_set.at[index, 'Age'] = ageMean[len(titles)][2]   # insert average age of passengers

    # Assign age grouping
    if (row['Age'] > 0 and row['Age'] < 21):
      df_set.at[index, 'AgeGroup'] = 1
    elif (row['Age'] < 41):
      df_set.at[index, 'AgeGroup'] = 2
    elif (row['Age'] < 61):
      df_set.at[index, 'AgeGroup'] = 3
    elif (row['Age'] > 60):
      df_set.at[index, 'AgeGroup'] = 4
    else:
      df_set.at[index, 'AgeGroup'] = 0                      # invalid age, e.g., 0 or -1


# Add column for categorical numeric feature when combining NumSibilingSpouse 
# and NumParentChild
# df_set refers to train_df or test_df
def CombineRelatives(df_set: pd.DataFrame):
  if not {'NumRelativeBrought'}.issubset(df_set.columns):   # Prevent column duplication
    df_set.insert(11,"NumRelativeBrought", 0, allow_duplicates=bool)

  for index, row in df_set.iterrows():
    sum = row['NumSiblingSpouse'] + row['NumParentChild']
    if (sum == 0):
      df_set.at[index, 'NumRelativeBrought'] = 0
    elif (sum < 4):
      df_set.at[index, 'NumRelativeBrought'] = 1
    else:
      df_set.at[index, 'NumRelativeBrought'] = 2
  # Change column type
  ToTypeInt(df_set, 'NumRelativeBrought')




###### ML & Visuals Functions
# Drop the columns before modelling
def DropColumn(df_set: pd.DataFrame, columns: object):
  df_set.drop(columns, axis=1, inplace=True)


# Displays heatmap for characteristic correlation
# Darker colors indicate stronger correlations, while lighter colors indicate 
# weaker correlations.
# Positive correlations (when one variable increases, the other variable tends to increase).
# Negative correlations (when one variable increases, the other variable tends to decrease).
def corr(data):
  plt.figure(figsize=(10, 10))
  correlation = data.corr(numeric_only=True)
  sns.heatmap(correlation, annot=True, cbar=True, cmap="RdYlGn")
  #plt.show()


# PerformanceMetric (includes Accuracy, Precision, Recall, and more) of 
# the trained ML model, includes confusion matrix
# testPredictions refer to the 1D array predictions of the test_df
# ModelName refers to the model in use
def PerformanceMetric(testPredictions, ModelName: object, testActual_df: pd.DataFrame):
  # F1 scoring
  accuracy = round(accuracy_score(testActual_df, testPredictions), 4)
  precision = round(precision_score(testActual_df, testPredictions), 4)
  recall = round(recall_score(testActual_df, testPredictions), 4)
  F1 = round(f1_score(testActual_df, testPredictions), 4)

  # Mean Absolute Error, Mean Squared Error, Root Mean Squared Error
  MAE = round(mean_absolute_error(testActual_df, testPredictions), 4)
  MSE = round(mean_squared_error(testActual_df, testPredictions), 4)
  RMSE = round(sqrt(MSE), 4)

  # Print values
  # print(ModelName, "Evaluation Metrics")
  # print("Accuracy:", accuracy)
  # print("Precision:", precision)
  # print("Recall:", recall)
  # print("F1 score:", F1)
  # print("Mean Absolute Error:", MAE)
  # print("Mean Squared Error:", MSE)
  # print("Root Mean Squared Error:", RMSE)

  metrics = pd.DataFrame({
    'Metric': ['F1 Score','Accuracy','Precision','Recall',
               'Mean Absolute Error', 'Mean Squared Error',
               'Root Mean Squared Error'], 
    'Value': [F1, accuracy, precision, recall, MAE, MSE, RMSE]
  })

  # Adjusting the parameters of the subplots figure
  fig = plt.figure(figsize=(14, 10))
  grid = gridspec.GridSpec(2, 1, left=0.1, bottom=0.1,
                           right=0.9, top=0.9, 
                           wspace=0.1, hspace=0.5)
  ax1 = plt.subplot(grid[0])
  ax1.bar(metrics['Metric'], metrics['Value'], color='skyblue')
  ax1.set_xlabel('Performance Metrics',fontsize=12)
  ax1.set_ylabel('Values',fontsize=12)
  ax1.set_title('{} Evaluation Metrics'.format(ModelName), fontsize=12)
  for i, value in enumerate(metrics['Value']):
    ax1.text(i, value/2, f'{value:.4f}', ha='center', va='bottom', 
             fontsize=12, color="#FF0000")
  

  # Displays the confusion matrix of the trained ML model
  ax2 = plt.subplot(grid[1])
  cm = confusion_matrix(testActual_df, testPredictions)    
  sns.heatmap(cm, annot=True, cmap="crest",
            xticklabels=['Negative','Positive'],
            yticklabels=['False','True'])
  ax2.set_ylabel('Actual',fontsize=12)
  ax2.set_xlabel('Prediction',fontsize=12)
  ax2.set_title('{} Confusion Matrix'.format(ModelName), fontsize=12)
  #plt.show()
  

# Calculates the probability of the column against Survived in the dataframe
# and plots the data
# df_set refers to the dataframe
# column refers to the characteristic 
def ProbabilityCheck(df_set: pd.DataFrame, column):
  result = df_set[[column, 'Survived']].groupby([column]).mean().sort_values(by='Survived', ascending=True)
  plt.figure(figsize=(10,6)) 
  plt.bar(result.index, result['Survived']) 
  plt.xlabel('{}'.format(column)) 
  plt.ylabel('Survival Probability') 
  plt.title('Survival Probability for {}'.format(column)) 
  plt.xticks(rotation=45) 
  
  for index, row in result.iterrows():
    plt.text(index, (row['Survived'])/2, round(row['Survived'], 2), 
             ha='center', va='bottom', fontsize=13, color="#FF7400")
  #plt.show()


# For non truncated display of datafames in the output for exe 
# df_set refers to the dataframe
# dataset_name refers to the name of the dataset 
def NonTruncDisplay(df_set: pd.DataFrame, dataset_name: object):
  # Prevent truncation of output for exe
  with pd.option_context('display.max_colwidth', None,
                        'display.max_rows', 15):
      print("\n {} Set:\n".format(dataset_name))
      print(df_set)
      pd.reset_option('display.max_colwidth')
      pd.reset_option('display.max_rows')




###### CLI Functions
# Convert datatypes to the correct types.
# Parameters:
# - df_set: the datafame 
# Returns:
# - df_set, the df with converted datatypes
def type_conversion(df_set):
# Correct data types after DataFrame creation
  df_set = df_set.astype({
      'Passenger ID': np.int64,
      'Ticket Class': np.int64,
      'Ticket Number': str,
      'Cabin': str,
      # 'Passenger Fare': float,
      'Embarkation Country': str,
      'Name': str,
      'Age': float,
      'Gender': str,
      'NumSiblingSpouse': np.int64,
      'NumParentChild': np.int64,
      'Survived': str
  })
  return df_set


# Validate user input based on a validation function and return the input if valid.
# Parameters:
# - prompt (str): The message to display to the user.
# - validation_func (function): A function that validates the user input.
# - max_attempts (int): The maximum number of attempts allowed before giving up.
# Returns:
# - str or None: The validated user input or None if the user entered 'back' or 
# failed validation.
def validate_input(prompt, validation_func, max_attempts=4):
  for i in range(max_attempts):
    # prints suggested entries for some characteristics
    if validation_func == is_valid_ticketclass:
      print("Only acceptable entries are '1', '2', and '3'")
    elif validation_func == is_valid_cabin:
      print("Suggested acceptable entry conventions are 'C123 C231', ", end='')
      print("'S/F123', 'C 123', 'C / E 123'")
    elif validation_func == is_valid_country:
      print("Suggested acceptable entry conventions are 'S', 'c'")
    elif validation_func == is_valid_gender:
      print("Only acceptable entries are 'male' or 'female'")
    elif validation_func == is_valid_survived:
      print("Only acceptable entries are 'Yes' or 'No'")
    user_input = input(prompt)
    if user_input.lower() == 'back':
      return None
    # if validation_func return is True, this data can be
    # inserted into dataframe
    if validation_func(user_input):
      return user_input
    print("Try Again, left {} attempts".format(max_attempts-i-1))
  return None


# Check if a string can be converted to a positive integer. For 
# NumSibilingSpouse and NumParentChild
# Parameters:
# - input_str (str): The input string to check.
# Returns:
# - bool: True if the input can be converted to an positivve 
# integer, False otherwise.
def is_int(input_str):
  try:
    return int(input_str) > -1
  except ValueError:
    return False


# Check if a string can be converted to a floating-point number.
# For Passenger Fare and Age
# Parameters:
# - input_str (str): The input string to check.
# Returns:
# - bool: True if the input can be converted to a float, False otherwise.
def is_float(input_str):
  try:
    return float(input_str) > 0.0
  except ValueError:
    return False


# Check if a string can be converted to an positive integer of range
# 1 to 3.
# Parameters:
# - input_str (str): The input string to check.
# Returns:
# - bool: True if the input can be converted to an positivve 
# integer, False otherwise.
def is_valid_ticketclass(input_str):
  try:
    # Allowed values are 1, 2, 3
    return int(input_str) > 0 and int(input_str) < 4
  except ValueError:
    return False


# Check if a string contains only ASCII characters between decimal values 32 and 122.
# Parameters:
# - input_str (str): The input string to check.
# Returns:
# - bool: True if the input contains valid characters, False otherwise.
def is_valid_name(input_str):
  if not input_str:
    return False
  # Only accepts ASCII from ! to z; some valid names are weird, no special
  # characters, numbers included, incase is "King Harri the 1st"
  return all(32 <= ord(char) <= 122 for char in input_str)


# Check if a cabin string follows the specified format rules.
# Parameters:
# - input_str (str): The cabin string to check.
# Returns:
# - bool: True if the cabin string is valid, False otherwise.
def is_valid_cabin(input_str):
  if not input_str:
    return False
  if not input_str[0].isalpha():
    return False
  # Not capital letter
  if not (ord(input_str[0])>=65 and ord(input_str[0])<=90):
    return False
  return all(32 <= ord(char) <= 90 for char in input_str)
  #return all(char.isalpha() or char.isdigit() for char in input_str)
  

# Check if a country string follows the specified format rules.
# Parameters:
# - input_str (str): The country string to check.
# Returns:
# - bool: True if the country string is valid, False otherwise.
def is_valid_country(input_str):
  return input_str.isalpha() and len(input_str) == 1


# Check if a gender string is 'male' or 'female'.
# Parameters:
# - input_str (str): The gender string to check.
# Returns:
# - bool: True if the gender string is 'male' or 'female', False otherwise.
def is_valid_gender(input_str):
  return input_str.lower() in ['male', 'female']


# Check if a survival string is 'Yes' or 'No'.
# Parameters:
# - input_str (str): The survival string to check.
# Returns:
# - bool: True if the survival string is 'Yes' or 'No', False otherwise.
def is_valid_survived(input_str):
  # if input is 'Yes' or 'No', return True
  return True if input_str == 'Yes' else True if input_str == 'No' else False


# Add a new entry to a Pandas DataFrame with specified 
# characteristics for a passenger.
# Parameters:
# - df (DataFrame): The Pandas DataFrame where the new entry will be added.
# - passenger_id (int): The unique identifier for the new passenger.
# - index_num (int): The index at which the new entry will be inserted 
# in the DataFrame.
# Returns:
# - DataFrame or None: The modified DataFrame with the new entry if 
# valid, or None if the entry is not added.
def input_new_entry(df, passenger_id, index_num):
  # Dictionary to map column names to their validation functions
  columns = {
    "Ticket Class": is_valid_ticketclass,
    "Ticket Number": is_valid_name,
    "Cabin": is_valid_cabin,
    "Passenger Fare": is_float,
    "Embarkation Country": is_valid_country,
    "Name": is_valid_name,
    "Age": is_float,
    "Gender": is_valid_gender,
    "NumSiblingSpouse": is_int,
    "NumParentChild": is_int,
    "Survived": is_valid_survived
  }
  for col, validation_func in columns.items():
      # prints Enter "characteristic" and valid entry type
      print(f"Enter {col} ({'int' if (validation_func == is_int or validation_func == is_valid_ticketclass) else 'float' if validation_func == is_float else 'string'}):")
      user_input = validate_input('', validation_func)
      if user_input is None:
        # invalid or user chose to go back
        try:
          df = df.drop(index_num, axis=0, inplace=True)
          return None
        except:
          return None
      else:
        # Ensures input is of correct type
        if validation_func == is_int or validation_func == is_valid_ticketclass:
          df.loc[index_num, col] = int(user_input)
        elif validation_func == is_float:
          df.loc[index_num, col] = float(user_input)
        else:       
          # locates [row, column]; 
          # [:2, col_name] = rows 0 to 2, 0 to col_name (for printing)
          df.loc[index_num, col] = user_input

  df.at[index_num, "Passenger ID"] = passenger_id
  df = type_conversion(df)
  print("Entry added successfully.\n")
  return df


# Concats 2 dataframes together (test_df and df -> new entries) 
# Parameters:
# - df (DataFrame): The Pandas DataFrame where the new entry are.
# - test_df (DataFrame): The Pandas DataFrame of existing entries.
# Returns:
# - df_combined: Concat DataFrame of the two arguments
def concat_new_entries(df: pd.DataFrame, test_df: pd.DataFrame): 
  df = type_conversion(df)
  frames = [test_df, df]
  df_combined = pd.concat(frames)
  # Preparing csv files for new data
  df_combined.to_csv("MS_1_Scenario_test_combined.csv",
                    index=False, encoding='utf-8')
  return df_combined