import os
import shutil

# --- Configuration ---
# This is the top-level directory containing your p1, p2, etc. folders
plots_folder = "plots_rows_deleted" 
subfolders_list = ["p1", "p2", "p3", "p4", "p5", "p6"]

def organize_plots():
    # 1. Loop through each subfolder in your list
    for sub in subfolders_list:
        # Construct the full path: e.g., "plots_rows_deleted/p1"
        current_base_path = os.path.join(plots_folder, sub)
        
        # Check if the subfolder exists before proceeding
        if not os.path.exists(current_base_path):
            print(f"Skipping {sub}: Folder not found.")
            continue
            
        print(f"Processing folder: {sub}...")

        # 2. Iterate through the files inside that specific subfolder
        for filename in os.listdir(current_base_path):
            file_path = os.path.join(current_base_path, filename)
            
            # Only process files, ignore folders already created
            if os.path.isfile(file_path):
                
                # 3. Split name (p1_bek_123.png -> ['p1', 'bek', '123.png'])
                parts = filename.split('_')
                
                if len(parts) >= 2:
                    category = parts[1]  # Extracts 'bek' or 'ben'
                    
                    # 4. Define the category folder path (e.g., plots_rows_deleted/p1/bek)
                    target_folder = os.path.join(current_base_path, category)
                    
                    # Create category folder if it doesn't exist
                    os.makedirs(target_folder, exist_ok=True)
                    
                    # 5. Move the file
                    dst_path = os.path.join(target_folder, filename)
                    shutil.move(file_path, dst_path)

    print("All folders organized successfully!")

if __name__ == "__main__":
    organize_plots()