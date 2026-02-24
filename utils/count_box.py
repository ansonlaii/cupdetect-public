import os
import glob

def count_boxes():
    # Path to labels directory
    labels_dir = 'labels'
    
    # Initialize counters
    total_class0 = 0
    total_class1 = 0
    files_with_boxes = 0
    
    # Get all text files in labels directory
    label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
    
    print("\nAnalyzing label files...")
    
    for file_path in label_files:
        file_class0 = 0
        file_class1 = 0
        
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    class_id = int(line.split()[0])
                    if class_id == 0:
                        file_class0 += 1
                        total_class0 += 1
                    elif class_id == 1:
                        file_class1 += 1
                        total_class1 += 1
                except (IndexError, ValueError):
                    print(f"Warning: Invalid format in {os.path.basename(file_path)}")
                    continue
        
        if file_class0 > 0 or file_class1 > 0:
            files_with_boxes += 1
            print(f"{os.path.basename(file_path)}: Class 0: {file_class0}, Class 1: {file_class1}")
    
    print("\nSummary:")
    print(f"Total files with boxes: {files_with_boxes}")
    print(f"Total Class 0 boxes: {total_class0}")
    print(f"Total Class 1 boxes: {total_class1}")
    print(f"Total boxes: {total_class0 + total_class1}")

if __name__ == "__main__":
    count_boxes()
