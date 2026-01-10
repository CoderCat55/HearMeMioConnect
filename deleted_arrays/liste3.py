import os
from pathlib import Path

# ===========================================
# CONFIGURATION - Change these as needed
# ===========================================
# The folder you want to scan (get files from)
FOLDER_TO_SCAN = r"C:\Users\Bemol\Documents\GitHub\HearMeMioConnect\rawgridplot\p2"

# Where to save the file list (ALWAYS save here)
SAVE_FOLDER = r"C:\Users\Bemol\Documents\hm_isef_docs"
SAVE_FILENAME = "file_listparticipant2.txt"

# Prefix to remove from filenames (set to "" to keep as-is)
PREFIX_TO_REMOVE = "p2_" 

# Extension to remove (e.g., ".png")
EXTENSION_TO_REMOVE = ".png"
# ===========================================

def get_all_files(folder_path):
    """Get all file names from a folder and its subfolders"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Error: Folder '{folder_path}' does not exist!")
        return [], 0
    
    print(f"📁 Scanning: {folder_path}")
    files = []
    total_folders = 0
    
    for item in folder.rglob("*"):
        if item.is_file():
            files.append(item.name)
        else:
            total_folders += 1
    
    return files, total_folders

def clean_filename(filename):
    """Removes prefix and the .png extension"""
    # 1. Remove Prefix
    if PREFIX_TO_REMOVE and filename.startswith(PREFIX_TO_REMOVE):
        filename = filename[len(PREFIX_TO_REMOVE):]
    
    # 2. Remove Extension (specifically .png)
    if filename.lower().endswith(EXTENSION_TO_REMOVE):
        filename = filename[:-len(EXTENSION_TO_REMOVE)]
        
    return filename

def save_file_list(file_list, save_path):
    """Save the file list to a text file"""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"FILE LIST\n")
            f.write(f"SCANNED FOLDER: {FOLDER_TO_SCAN}\n")
            f.write(f"PREFIX REMOVED: '{PREFIX_TO_REMOVE}'\n")
            f.write(f"EXTENSION REMOVED: '{EXTENSION_TO_REMOVE}'\n")
            f.write(f"TOTAL FILES: {len(file_list)}\n")
            f.write("=" * 60 + "\n\n")
            
            for i, filename in enumerate(sorted(file_list), 1):
                clean_name = clean_filename(filename)
                f.write(f"{i:4}. {clean_name}\n")
        
        print(f"✅ File list saved to: {save_path}")
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False

def main():
    print("=" * 60)
    print("CLEAN FILE LIST GENERATOR")
    print("=" * 60)
    
    save_path = os.path.join(SAVE_FOLDER, SAVE_FILENAME)
    files, folder_count = get_all_files(FOLDER_TO_SCAN)
    
    if not files:
        print("\n⚠️  No files found.")
        return

    if save_file_list(files, save_path):
        print(f"\n3️⃣  Preview of cleaned files (first 10):")
        print("-" * 40)
        for i, filename in enumerate(sorted(files)[:10], 1):
            print(f"{i:2}. {clean_filename(filename)}")
        
        try:
            print(f"\n📂 Opening saved file...")
            os.startfile(save_path)
        except:
            pass

    print("\nDone! ✨")

if __name__ == "__main__":
    main()