import os
import glob
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def evaluate_original(model_path, images_dir, conf_thres=0.50):
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = YOLO(model_path)
    
    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    
    if not image_paths:
        print(f"No images found in {images_dir}")
        return
        
    print(f"Found {len(image_paths)} total defective images in original dataset.")
    
    # Track statistics
    total_defective_images = len(image_paths)
    metric1_success = 0 # YOLO + Secondary
    metric2_success = 0 # YOLO + Primary + Secondary
    
    print("\nStarting evaluation on ORIGINAL Defective images...")
    
    for img_path in image_paths:
        # Predict
        results = model.predict(source=img_path, conf=conf_thres, verbose=False)
        
        pred_classes = set()
        for box in results[0].boxes:
            pred_classes.add(int(box.cls[0]))
        
        success_m1 = 2 in pred_classes
        success_m2 = (1 in pred_classes) and (2 in pred_classes)
        
        if success_m1:
            metric1_success += 1
        if success_m2:
            metric2_success += 1
                
    m1_acc = (metric1_success / total_defective_images) * 100 if total_defective_images > 0 else 0
    m2_acc = (metric2_success / total_defective_images) * 100 if total_defective_images > 0 else 0
    
    print("\n" + "="*60)
    print("ORIGINAL DATA ACCURACY EVALUATION REPORT")
    print("="*60)
    print(f"Folder: {images_dir}")
    print(f"Total Defective Images Evaluated: {total_defective_images}")
    print("-" * 60)
    
    print(f"Metric 1: YOLO + Secondary Label Only")
    print(f"  Condition: Model predicted at least one Defect Region (Class 2)")
    print(f"  Success:   {metric1_success} / {total_defective_images} -> [{m1_acc:.2f}% Accuracy]")
    print()
    print(f"Metric 2: YOLO + Primary + Secondary Labels")
    print(f"  Condition: Model predicted BOTH Defective Insulator (Class 1) AND Defect Region (Class 2)")
    print(f"  Success:   {metric2_success} / {total_defective_images} -> [{m2_acc:.2f}% Accuracy]")
    
    # Generate and save visualization
    if total_defective_images > 0:
        plt.figure(figsize=(10, 6))
        metrics = ['YOLO + Secondary\n(Defect Region Only)', 'YOLO + Primary + Secondary\n(Both Insulator & Defect)']
        accuracies = [m1_acc, m2_acc]
        
        bars = plt.bar(metrics, accuracies, color=['#9C27B0', '#FF9800'])
        plt.ylim(0, 110) # 0 to 110% for head room
        plt.ylabel('Accuracy (%)')
        plt.title(f'Original Data Defect Detection Accuracy\n(Total Images: {total_defective_images}, Conf: {conf_thres})')
        
        # Add text labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
            
        plot_path = 'original_data_accuracy_chart.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved visual chart to: {plot_path}")

if __name__ == '__main__':
    weights = './runs/detect/train_v2/weights/best.pt'
    images_dir = '/Users/maheshvellogi/Downloads/Insulator_Project/original_data/Defective_Insulators/images'
    evaluate_original(weights, images_dir)
