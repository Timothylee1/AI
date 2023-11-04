from HeaderFile import *
# from EDA import *
# from ML import *
# Run one Python file from another; prevents variable shadowing 
# and undefined / undesirable events (e.g., suddenly import EDA does not work)
import subprocess 

CLI_test_df = pd.DataFrame() 

if os.path.exists("MS_1_Scenario_test_combined.csv"):
    # Delete the file
    os.remove("MS_1_Scenario_test_combined.csv")
    print("Removed 'MS_1_Scenario_test_combined.csv' file")

def main():
  global CLI_test_df 
  # data = {
  #     'Passenger ID' : [1],
  #     'Passenger Fare' : [100.1],
  #     'Ticket Class' : [2],
  #     'Ticket Number' : ['312313'],
  #     'Cabin' : ['C123'],
  #     'Embarkation Country' : ['S'],
  #     'Name' : ['John, Mr.'],
  #     'Age' : [20.3],
  #     'Gender' : ['male'],
  #     'NumSiblingSpouse' : [2],
  #     'NumParentChild' : [2],
  #     'Survived' : ['Yes']
  # }
  data = ['Passenger ID', 'Passenger Fare', 'Ticket Class','Ticket Number',
          'Cabin', 'Embarkation Country', 'Name', 'Age',
          'Gender', 'NumSiblingSpouse','NumParentChild', 'Survived']
  
  df = pd.DataFrame(columns=data)
  while CLI_test_df.empty:
    subprocess.run(["python", "File_Upload_Widget.py"]) 
    ## Reading CSV files
    train_df, CLI_test_df = path_to_csv()
    if not CLI_test_df.empty:
      # Read the dataset. The imported CSV is called a DataFrame --> df.
      ## Passenger Fare filtering (Removes '$', round to 2 d.p., cast type as float)
      RemoveDollarSign(CLI_test_df)
      # print(train_df.head())
    else:
      print("Unable to retreive csv files...")

  # df.info()
  # print(df)
  # locates the last value in Passenger ID
  passenger_id = CLI_test_df['Passenger ID'].iloc[-1] + 1

  while True:
    print("\nMain Menu:")
    print("1. Input new entry")
    print("2. Display NEW test entries")
    print("3. Display ALL test entries")
    print("4. Perform EDA")
    print("5. Perform ML")
    print("6. Display visuals for EDA / ML")
    print("7. Insurance Menu")
    print("8. Exit program")
    option = input("Select an option (1/8): ")
  
    do_i_combine = False
    try:
      index_num = df.index[-1] + 1
      do_i_combine = True
      df = type_conversion(df)
    except:
      index_num = CLI_test_df.index[-1] + 1

    # Input new entry
    if option == '1': 
      # end='' removes the default '\n' as the parameter at the end of the print
      print("Please enter 'back' to return to Main page, ", end='')
      print("any time throughout the entry process. Doing so will ", end='')
      print("the entry.")
      new_df = input_new_entry(df, passenger_id, index_num)
      if new_df is not None:
        df = type_conversion(new_df)
        passenger_id += 1
      else:
        print("Entries not recorded, proceeding back to Main Menu...\n")

    # Display new entry
    elif option == '2':
      print(df,"\n")

    # Display all entries (including exisitng data)
    elif option == '3':
      # Combine 
      if do_i_combine:
        print(concat_new_entries(df, CLI_test_df), '\n')
        print("MS_1_Scenario_test_combined.csv created / updated\n")
      else:
        print(CLI_test_df, '\n')

    # Run EDA
    elif option == '4':
      if do_i_combine:
        concat_new_entries(df, CLI_test_df)
        print("MS_1_Scenario_test_combined.csv created / updated\n")

      subprocess.run(["python", "EDA.py"]) # executes EDA.py script

    # Run ML
    elif option == '5':
      if do_i_combine:
        concat_new_entries(df, CLI_test_df)
        print("MS_1_Scenario_test_combined.csv created / updated\n")

      subprocess.run(["python", "ML.py"])  # executes ML.py script

    # Run EDA & ML Visuals
    elif option == '6':
      print("")
      subprocess.run(["python", "Visuals.py"])  # executes Visuals.py script

    # Run insurance subprocess
    elif option == '7':
      print("")
      subprocess.run(["python", "Insurance.py"])  # executes Insurance.py script

    elif option == '8':
      break
    else:
      print("Invalid option selected...\n")

if __name__ == "__main__":
  main()
