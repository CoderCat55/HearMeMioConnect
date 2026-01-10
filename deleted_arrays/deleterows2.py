"""
1st coulmn consist of order number + filename
2nd column for rest1
3rd coulmn for rest2 
4rd coulmn is irrelevant

Aim: delete the specified coulmn numbers from npy files inside the main folder and save them into output folder.Also in order to see which rows we have deleted create deletedrow arrays

# --- Constants ---
EXCEL_PATH = r""
MAIN_FOLDER_PATH = r""  #where our original .npy files are stored
OUTPUT_FOLDER_PATH = r""  #the new folder that all npy arrays will be stored
TEXT_LOG_NAME = "px.txt"
TEXT_LOG_PATH = r""

Algorithm:
1- Get excel file 
2- read column 1 and do  column1 name cleaning:
3- create an array named "rowdelete{filename}"
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
7- save these arrays into a text file name "TEXT_LOG_NAME" to location "TEXT_LOG_PATH"
8- open each npy file delete the rows according to arrays stored.
9- save these npy files inside "OUTPUT_FOLDER_PATH"

Note :
if the row is left skip the row  
if rest1 or rest2 left blank add to rowdelete as none

column1 name cleaning:
delete the order number before the filename  (for file 12. bek_1232411 remove 12.)
        clean_filename = re.sub(r'^\d+\.\s*', '', raw_filename)
        clean_filename = clean_filename.replace('grid_', '').strip()
        clean_filename = clean_filename.replace('.png', '').strip()
        clean_filename = clean_filename.replace('p1_', '').strip()
        clean_filename = clean_filename.replace('p2_', '').strip()
        clean_filename = clean_filename.replace('p3_', '').strip()
        clean_filename = clean_filename.replace('p4_', '').strip()
        clean_filename = clean_filename.replace('p5_', '').strip()
        clean_filename = clean_filename.replace('p6_', '').strip()
        clean_filename = clean_filename.replace('_missing_grid', '').strip()
        clean_filename = clean_filename.replace('.npy', '').strip()

""" 
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path

# --- Constants ---
EXCEL_PATH = r"C:\Users\Bemol\Documents\GitHub\HearMeMioConnect\deleted_arrays\croppeddata (6) - Kopya.xlsx"
MAIN_FOLDER_PATH = r"C:\Users\Bemol\Documents\GitHub\HearMeMioConnect\calibration_data\p6"  # where our original .npy files are stored
OUTPUT_FOLDER_PATH = r"C:\Users\Bemol\Documents\GitHub\HearMeMioConnect\rows_deleted\p6"  # the new folder that all npy arrays will be stored
TEXT_LOG_NAME = "p6deletedrowsarray.txt"
TEXT_LOG_PATH = r"C:\Users\Bemol\Documents\GitHub\HearMeMioConnect\deleted_arrays"

def clean_filename(raw_filename):
    """Clean the filename according to specified rules."""
    if pd.isna(raw_filename) or raw_filename == '':
        return None
    
    raw_filename = str(raw_filename).strip()
    
    # Remove order number (e.g., "12. " from "12. bek_1232411")
    clean_name = re.sub(r'^\d+\.?\s*', '', raw_filename)
    
    # Remove all specified prefixes/suffixes
    replacements = [
        'grid_', '.png', 'p1_', 'p2_', 'p3_', 
        'p4_', 'p5_', 'p6_', '_missing_grid', '.npy'
    ]
    
    for pattern in replacements:
        clean_name = clean_name.replace(pattern, '').strip()
    
    # Remove any leading dots and spaces that might remain
    clean_name = clean_name.lstrip('. ')
    
    return clean_name


def process_excel_and_create_delete_arrays(excel_path, main_folder_path):
    """
    Process Excel file and create row deletion arrays for each NPY file.
    
    Returns:
        dict: Dictionary mapping filenames to arrays of row indices to delete
    """
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Dictionary to store deletion arrays
    deletion_dict = {}
    
    # Process each row
    for idx, row in df.iterrows():
        # Step 2: Get and clean filename from column 1 (index 0)
        raw_filename = row.iloc[0]
        
        # Skip if row is empty
        if pd.isna(raw_filename) or str(raw_filename).strip() == '':
            continue
        
        # Step 3: Clean filename and create deletion array
        filename = clean_filename(raw_filename)
        if filename is None:
            continue
        
        rowdelete = []
        
        # Step 4: Process rest1 (column 2, index 1)
        rest1_value = row.iloc[1]
        
        if pd.isna(rest1_value) or str(rest1_value).strip().lower() == 'x':
            rest1 = False
        else:
            rest1 = True
            try:
                x = int(rest1_value)
                # Calculate all integers from 0 to x (inclusive)
                rowdelete.extend(range(0, x + 1))
            except (ValueError, TypeError):
                # If conversion fails, treat as None
                rowdelete.append(None)
        
        # Step 5: Process rest2 (column 3, index 2)
        rest2_value = row.iloc[2]
        
        if pd.isna(rest2_value) or str(rest2_value).strip().lower() == 'x':
            rest2 = False
        else:
            rest2 = True
            try:
                x = int(rest2_value)
                # Get the actual file length
                npy_file_path = os.path.join(main_folder_path, f"{filename}.npy")
                
                if os.path.exists(npy_file_path):
                    arr = np.load(npy_file_path)
                    file_length = len(arr)
                    # Calculate all integers from x to file_length (inclusive)
                    rowdelete.extend(range(x, file_length))
                else:
                    print(f"Warning: File not found - {npy_file_path}")
                    rowdelete.append(None)
            except (ValueError, TypeError):
                # If conversion fails, treat as None
                rowdelete.append(None)
        
        # Remove duplicates and sort (exclude None values for sorting)
        valid_indices = [i for i in rowdelete if i is not None]
        none_count = rowdelete.count(None)
        
        if valid_indices:
            rowdelete = sorted(set(valid_indices))
            # Add None values back if they existed
            rowdelete.extend([None] * none_count)
        
        deletion_dict[filename] = rowdelete
    
    return deletion_dict


def save_deletion_log(deletion_dict, log_path, log_name):
    """Save deletion arrays to a text file."""
    full_log_path = os.path.join(log_path, log_name)
    
    with open(full_log_path, 'w') as f:
        for filename, delete_rows in deletion_dict.items():
            f.write(f"rowdelete{filename} = {delete_rows}\n")
    
    print(f"Deletion log saved to: {full_log_path}")


def process_npy_files(deletion_dict, main_folder_path, output_folder_path):
    """Delete specified rows from NPY files and save to output folder."""
    # Create output folder if it doesn't exist
    Path(output_folder_path).mkdir(parents=True, exist_ok=True)
    
    for filename, delete_rows in deletion_dict.items():
        npy_file_path = os.path.join(main_folder_path, f"{filename}.npy")
        
        if not os.path.exists(npy_file_path):
            print(f"Skipping - file not found: {npy_file_path}")
            continue
        
        # Load the NPY file
        arr = np.load(npy_file_path)
        
        # Filter out None values from delete_rows
        valid_delete_rows = [i for i in delete_rows if i is not None]
        
        if valid_delete_rows:
            # Delete specified rows
            arr_filtered = np.delete(arr, valid_delete_rows, axis=0)
        else:
            arr_filtered = arr
        
        # Save to output folder
        output_path = os.path.join(output_folder_path, f"{filename}.npy")
        np.save(output_path, arr_filtered)
        
        print(f"Processed: {filename}.npy - Deleted {len(valid_delete_rows)} rows")


def main():
    """Main execution function."""
    print("Starting NPY file processing...")
    
    # Validate paths
    if not os.path.exists(EXCEL_PATH):
        print(f"Error: Excel file not found at {EXCEL_PATH}")
        return
    
    if not os.path.exists(MAIN_FOLDER_PATH):
        print(f"Error: Main folder not found at {MAIN_FOLDER_PATH}")
        return
    
    # Step 1-6: Process Excel and create deletion arrays
    print("\nStep 1-6: Processing Excel file...")
    deletion_dict = process_excel_and_create_delete_arrays(EXCEL_PATH, MAIN_FOLDER_PATH)
    
    print(f"Found {len(deletion_dict)} files to process")
    
    # Step 7: Save deletion log
    print("\nStep 7: Saving deletion log...")
    save_deletion_log(deletion_dict, TEXT_LOG_PATH, TEXT_LOG_NAME)
    
    # Step 8-9: Process NPY files
    print("\nStep 8-9: Processing NPY files...")
    process_npy_files(deletion_dict, MAIN_FOLDER_PATH, OUTPUT_FOLDER_PATH)
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()