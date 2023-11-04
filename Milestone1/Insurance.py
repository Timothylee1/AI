#!/usr/bin/env python3

## Libraries
import os
# These libraries are required to perform dataset manupulation.
import pandas as pd

def insurance(df: pd.DataFrame):
  #Allows display for all rows
  pd.set_option("display.max_rows", None)

  # Define function to calculate premium_charged for each row
  def calculate_premium(Survivability):
    if Survivability > 0.50:
      if Survivability == 1:
        return "$" + str(100)
      return "$" + str(round(100 + ((1-Survivability)*100),2) )
    else:
      return "Don't Insure"

  # Apply function to each row of 1DataFrame and create a new column 'premium_charged'
  df['premium_charged'] = df['Alive Probability'].apply(calculate_premium)

  # Print the premium_charged column
  #print(df[['Passenger ID', 'premium_charged']])

  while True:
    print("\nInsurance Menu:")
    print("1. Display the output")
    print("2. Export to a new CSV file")
    print("3. Exit")

    choice = input("Enter your choice (1,2,3): ")

    if choice == '1':
      # Display the 'Passenger ID' and 'premium_charged' columns
      print(df[['Passenger ID', 'premium_charged']])

    elif choice == '2':
      # Export the DataFrame to a new CSV file
      export_filename = input("Enter the export filename (e.g., output.csv): ")
      df.to_csv(export_filename, index=False)
      print(f"Data exported to {export_filename}")
      
    elif choice == '3':
      break
    else:
      print("Invalid choice. Please select 1, 2, or 3.")


def main():

  if os.path.exists("insurance_test_predictions.csv"):
    #reads these 2 selected columns
    df = pd.read_csv('insurance_test_predictions.csv', usecols=['Alive Probability','Passenger ID'])
    insurance(df)
  else:
    print("insurance_test_predictions.csv file not found!")
    print("Please run ML first\n")
    k=input("Input a character followed by pressing Enter to exit")     

if __name__ == "__main__":
  main()