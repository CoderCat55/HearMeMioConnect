
"""
1st coulmn consist of order number + filename
2nd column for rest1
3rd coulmn for rest2 
4rd coulmn is irrelevant

Aim: delete the specified coulmn numbers from npy files inside the main folder and save them into output folder.Also in order to see which rows we have deleted create deletedrow arrays

# --- Constants ---
EXCEL_PATH = r""
MAIN_FOLDER_PATH = r""
OUTPUT_FOLDER_PATH = r""
TEXT_LOG_NAME = "deneme1.txt"
TEXT_LOG_PATH = r""

Algorithm:
1- Get excel file 
2- read column 1, delete the order number before the filename  (for file 12. bek_1232411 remove 12.)
3- create an array named "rowdelete+{filename}"
4- read the column 2 
    if value of the cell = x
        rest1= false
    else
        rest1=True
        Calculate all integers from 0 to x , include 0 and x
        add these values to rowdelete array
5- read the column 3
    if value of the cell = x
        rest2= false
    else
        rest2=True
        Calculate all integers from x to length of the original file,include x and file length
        add these values to rowdelete array
6- Repeat steps 2,3,4,5 for each file
7- save these arrays into a text file name specified by the user to location also specified by user
8- open each npy file delete the rows according to arrays stored.
9- save files inside 

Note :
if the row is left skip the row  
if rest1 or rest2 left blank add to rowdelete as none


""" 
import pandas as pd
import numpy as np
import os
import re

# --- Constants ---
EXCEL_PATH = r"C:\Users\Bemol\Documents\GitHub\HearMeMioConnect\croppeddata (3) - Kopya.xlsx"
MAIN_FOLDER_PATH = r"C:\Users\Bemol\Documents\GitHub\HearMeMioConnect\calibration_data\p3"
OUTPUT_FOLDER_PATH = r"C:\Users\Bemol\Documents\GitHub\HearMeMioConnect\deletedrows\p3"
TEXT_LOG_PATH = r"C:\Users\Bemol\Documents\GitHub\HearMeMioConnect\deletedrows\deleted arrays\p3.txt"

def process_files():
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)

    # 1. Get excel file 
    df = pd.read_excel(EXCEL_PATH)
    
    log_entries = []

    for index, row in df.iterrows():
        # Skip empty rows
        if pd.isna(row.iloc[0]):
            continue

        # 2. Extract filename
        raw_filename = str(row.iloc[0])
        
        # Step A: Remove the leading order number and dot (e.g., "12. ")
        clean_filename = re.sub(r'^\d+\.\s*', '', raw_filename)
        
        # Step B: Remove "grid_" from anywhere in the filename
        clean_filename = clean_filename.replace('grid_', '').strip()
        
        # Step B: Remove "grid_" from anywhere in the filename
        clean_filename = clean_filename.replace('.png', '').strip()

        # Step B: Remove "grid_" from anywhere in the filename
        clean_filename = clean_filename.replace('p2_', '').strip()

        # Step B: Remove "grid_" from anywhere in the filename
        clean_filename = clean_filename.replace('p3_', '').strip()

        # Ensure it has the .npy extension
        npy_filename = clean_filename if clean_filename.endswith('.npy') else f"{clean_filename}.npy"
        full_input_path = os.path.join(MAIN_FOLDER_PATH, npy_filename)

        # 3. Initialize rowdelete list
        row_delete_indices = []

        # Check if file exists
        if not os.path.isfile(full_input_path):
            print(f"Warning: {npy_filename} not found. Skipping row {index + 1}.")
            continue

        # Load file to get length for range calculations
        data = np.load(full_input_path)
        file_length = len(data)

        # 4. Process Column 2 (rest1: delete 0 to x)
        val1 = row.iloc[1]
        if pd.isna(val1):
            row_delete_indices.append("None")
        elif str(val1).lower() != 'x':
            try:
                limit1 = int(val1)
                # Inclusive: 0 to x
                row_delete_indices.extend(list(range(0, limit1 + 1)))
            except ValueError:
                pass

        # 5. Process Column 3 (rest2: delete x to end)
        val2 = row.iloc[2]
        if pd.isna(val2):
            row_delete_indices.append("None")
        elif str(val2).lower() != 'x':
            try:
                limit2 = int(val2)
                # Inclusive: x to file length
                row_delete_indices.extend(list(range(limit2, file_length)))
            except ValueError:
                pass

        # 6 & 7. Format log entry
        # Store a copy of the list (including "None") for the text log
        log_label = f"rowdelete+{clean_filename}"
        log_entries.append(f"{log_label}: {row_delete_indices}")

        # 8. Filter numeric indices for actual deletion
        numeric_indices = sorted(list(set([i for i in row_delete_indices if isinstance(i, int)])))
        valid_indices = [i for i in numeric_indices if i < file_length]
        
        # Delete rows
        modified_data = np.delete(data, valid_indices, axis=0)

        # 9. Save files
        output_save_path = os.path.join(OUTPUT_FOLDER_PATH, npy_filename)
        np.save(output_save_path, modified_data)
        print(f"Processed: {npy_filename}")

    # Write logs to text file
    with open(TEXT_LOG_PATH, 'w') as f:
        f.write("\n".join(log_entries))
    
    print(f"\nTask Finished. Log saved to: {TEXT_LOG_PATH}")

if __name__ == "__main__":
    process_files()