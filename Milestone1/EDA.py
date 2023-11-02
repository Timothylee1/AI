#!/usr/bin/env python
# Importing all functions defined in SetupHeader.py
from HeaderFile import *

print("EDA Job Starting...\n")

## Reading CSV files
# Read the dataset. The imported CSV is called a DataFrame --> df.
try:
    test_df = pd.read_csv('MS_1_Scenario_test_combined.csv', encoding='utf-8')
    print("Including new entries in the test set")
except:
  test_df = pd.read_csv('MS_1_Scenario_test.csv', encoding='utf-8') 

train_df = pd.read_csv('MS_1_Scenario_train.csv', encoding='utf-8')

# Combine 
# frames = [train_df, test_df]
# train_df = pd.concat(frames)
# print(train_df.info())
# print('_'*50)
# print(test_df.info())

## Passenger Fare filtering (Removes '$', round to 2 d.p., cast type as float)
RemoveDollarSign(train_df)
RemoveDollarSign(test_df)
# print(train_df.head())
# print(test_df.head())


# condition for new inputs?!
existing_df = train_df.copy()


## Data type discrepancy and applying numeric categorization where possible
# Round up the Age values
train_df['Age'] = train_df['Age'].apply(np.ceil)
test_df['Age'] = test_df['Age'].apply(np.ceil)

# Replace alphanumeric Cabin entries with the first alphabet present
CabinFilter(train_df)
CabinFilter(test_df)

# Binary conversion "Yes" and "male" to 1, "No" and "female" to 0
# Converting for train_df
Conversion(train_df, 'Survived', "Yes", "No")
Conversion(train_df, 'Gender', "male", "female")
# Converting for test_df
Conversion(test_df, 'Survived', "Yes", "No")
Conversion(test_df, 'Gender', "male", "female")

# Change DataFrame type from object to int
ToTypeInt(train_df, 'Survived')
ToTypeInt(train_df, 'Gender')
ToTypeInt(train_df, 'Age')
ToTypeInt(test_df, 'NumSiblingSpouse')
ToTypeInt(test_df, 'NumParentChild')
ToTypeInt(test_df, 'Ticket Class')
ToTypeInt(test_df, 'Survived')
ToTypeInt(test_df, 'Gender')
ToTypeInt(test_df, 'Age')
# print(train_df.info())
# print(test_df.info())
# Copies over dataframe before columns are dropped
train_df_B4mod = train_df.copy()
test_df_B4mod = test_df.copy()

## Embarkation Country
# Removes Embarkation Country entries that aren't 'A' to 'Z' or 'a' to 'z' 
# and categorizes the rest. Known countries: C, Q, S,  to 1, 2, 3. 
# Everything else is group 4
train_df = RemoveOutlier(train_df)
test_df = RemoveOutlier(test_df)
# print(train_df.head(n=131))         # Passenger ID 130 should be removed
# print(test_df.head(n=70))           # Passenger ID 865 row should be removed

## Name and Age
# list available titles [0, 1, 2, 3]
titles = ['Miss','Mrs', 'Mr', 'Master']
# Titles less than 10: Ms, Dr, Rev, Major 
# and Special titles will be enumerated <-- len(titles) 

# array to help find mean of each title's age 
# [0][0] titles index 0, sum of the ages; 
# [4][1] people with outside of the titles list, total number;
# [3][2] titles index 3, mean age of the specified title
ageMean = np.zeros((5, 3)).astype(int)

# Add titles into the Title column and check age from train_df
# Replaces "0" in Age to the mean of the name's respective title & groups them
# Check whether the count and average are correct; 
# (134 94 352 31  19) + 167 0s = 797.
# Results in 2 new columns, Title and AgeGroup, which will replace Name and Age 
# in the modelling stage
Amend_Title_Age(train_df, True, titles, ageMean)
Amend_Title_Age(test_df, False, titles, ageMean)

## NumSibilingSpouse and NumParentChild
# Values in these 2 columns will be combined (Sum) and categorised into 3 groups. 
# Sum = 0, group 0; 0 < Sum < 4, group 1; the rest are in group 2
CombineRelatives(train_df)
CombineRelatives(test_df)

# Copies over dataframe before columns are dropped
train_df_B4drop = train_df.copy()
test_df_B4drop = test_df.copy()

print("EDA Job Done")
# NonTruncDisplay(test_df, 'test')
# Prevents auto closure of command promt for exe
# k=input("Input a character followed by pressing Enter to exit") 