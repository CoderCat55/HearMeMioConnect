import os
import shutil

# --- Configuration (Must match your organization script) ---
plots_folder = "plots_rows_deleted" 
subfolders_list = ["p1", "p2", "p3", "p4", "p5", "p6"]

def undo_organization():
    for sub in subfolders_list:
        subfolder_path = os.path.join(plots_folder, sub)
        
        if not os.path.exists(subfolder_path):
            continue
            
        print(f"Flattening folder: {sub}...")

        # Loop through everything inside p1, p2, etc.
        for item in os.listdir(subfolder_path):
            item_path = os.path.join(subfolder_path, item)
            
            # If we find a directory (like 'bek' or 'ben'), we need to empty it
            if os.path.isdir(item_path):
                for filename in os.listdir(item_path):
                    file_src = os.path.join(item_path, filename)
                    file_dst = os.path.join(subfolder_path, filename)
                    
                    # Move file back up to the subfolder (p1, p2, etc.)
                    shutil.move(file_src, file_dst)
                
                # After moving all files out, delete the empty category folder
                os.rmdir(item_path)
                print(f"  - Removed category folder: {item}")

    print("Undo complete. Files are back in their original subfolders.")

if __name__ == "__main__":
    undo_organization()