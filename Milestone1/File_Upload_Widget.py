import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
import os


def update_file_path(filename, new_file_path):
  # Read the current content of the file into memory
  updated_lines = []

  with open("selected_file_paths.txt", "r") as file:
    for line in file:
      parts = line.strip().split("=")
      if len(parts) == 2:        
        current_filename = parts[0]
        file_path = parts[1]
        if current_filename == filename and file_path != new_file_path:
          updated_lines.append(f"{filename}={new_file_path}\n")
          # Write the updated content back to the file
          with open("selected_file_paths.txt", "w") as file:
            file.writelines(updated_lines)



def file_upload(event, filename):
  file_path = filedialog.askopenfilename(title="Select the {} CSV file".format(filename))

  # Check if the user canceled the file selection
  if not file_path:
    print("File not found.\n")
  else:
    with open("selected_file_paths.txt", "a") as file:
      file.write(f"{filename}={file_path}\n")
    update_file_path(filename, file_path)
    messagebox.showinfo("{} file Selected".format(filename), "File Uploaded")


def exit_app():
  root.destroy()


if os.path.exists("selected_file_paths.txt"):
  # Delete the file
  os.remove("selected_file_paths.txt")
# Ensure selected_file_paths.txt is created if it doesn't exist
if not os.path.exists("selected_file_paths.txt"):
  open("selected_file_paths.txt", "w").close()
  
root = tk.Tk()
root.title("CSV File Uploader")
label1 = tk.Label(root, text="Please upload the test and training csv files.")
label1.pack()
file_train_button = tk.Button(root, text="Train File Upload")
file_train_button.bind("<Button-1>", lambda event: file_upload(event, 'train'))
file_train_button.pack(pady=10)
file_test_button = tk.Button(root, text="Test File Upload")
file_test_button.bind("<Button-1>", lambda event: file_upload(event, 'test'))
file_test_button.pack(pady=10)
exit_button = tk.Button(root, text="Exit", command=exit_app)
exit_button.pack(pady=10)
root.mainloop()