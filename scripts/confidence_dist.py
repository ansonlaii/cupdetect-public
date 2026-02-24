import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd

# Load the model
model_path = r"C:\Users\laian\syde_proj\epoch500.pt"
model = YOLO(model_path)

# Path to images
images_dir = r"C:\Users\laian\syde_proj\QA"
image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Lists to store confidence scores and bounding box information
confidence_scores = []
bbox_data = []

# Create a directory for YOLOv5 format labels
yolo_labels_dir = 'yolo_labels'
os.makedirs(yolo_labels_dir, exist_ok=True)

# Process each image
for img_path in image_files:
    # Run inference
    results = model(img_path)
    
    # Get image dimensions for normalization
    img_height, img_width = results[0].orig_shape
    
    # Extract results
    for result in results:
        boxes = result.boxes
        
        # Prepare YOLOv5 format labels for this image
        yolo_labels = []
        
        # Extract confidence scores
        if len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf)
                confidence_scores.append(conf)
                
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls)
                
                # Add to bbox data
                bbox_data.append({
                    'image': os.path.basename(img_path),
                    'class': cls,
                    'confidence': conf,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })
                
                # Convert to YOLOv5 format (class x_center y_center width height)
                # All values normalized between 0 and 1
                x_center = (x1 + x2) / (2 * img_width)
                y_center = (y1 + y2) / (2 * img_height)
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Add to YOLOv5 labels
                yolo_labels.append(f"{cls} {x_center} {y_center} {width} {height}")
        
        # Save YOLOv5 format labels
        if yolo_labels:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(yolo_labels_dir, f"{base_name}.txt")
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

# After processing all images, plot the confidence score distribution
if confidence_scores:
    # Set environment variable to handle OpenMP runtime duplication issue
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidence_scores, bins=20, alpha=0.7, color='blue')
    plt.title('Distribution of Confidence Scores for REFINED model with diverse data', fontsize=20)
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_distribution.png')
    plt.close()
    
    # Print some statistics about confidence scores
    avg_conf = sum(confidence_scores) / len(confidence_scores)
    print(f"Average confidence score: {avg_conf:.4f}")
    print(f"Total detections: {len(confidence_scores)}")

