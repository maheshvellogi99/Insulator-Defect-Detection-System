import argparse
import os
import glob
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def load_ground_truth(label_path):
    classes = set()
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes.add(int(parts[0]))
    return classes

def evaluate(model_path, images_dir, labels_dir, conf_thres=0.25, save_failed=False, output_dir='runs/failed_predictions'):
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = YOLO(model_path)
    
    if save_failed:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Failed predictions will be saved to: {output_dir}")
        
    image_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    
    if not image_paths:
        print(f"No images found in {images_dir}")
        return
        
    print(f"Found {len(image_paths)} total test images.")
    
    # Track statistics
    total_defective_images = 0
    metric1_success = 0 # YOLO + Secondary
    metric2_success = 0 # YOLO + Primary + Secondary
    
    print("\nStarting evaluation...")
    
    for img_path in image_paths:
        base_name = os.path.basename(img_path)
        label_name = base_name.replace('.jpg', '.txt')
        label_path = os.path.join(labels_dir, label_name)
        
        gt_classes = load_ground_truth(label_path)
        
        # Test images that have a defect region (Class 2)
        if 2 in gt_classes:
            total_defective_images += 1
            
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
                
            if save_failed and (not success_m1 or not success_m2):
                # Visualize and save the ones that failed
                res_plotted = results[0].plot()
                out_path = os.path.join(output_dir, f"failed_conf{conf_thres}_{base_name}")
                cv2.imwrite(out_path, res_plotted)
                
    print("\n" + "="*60)
    print("CUSTOM ACCURACY EVALUATION REPORT")
    print("="*60)
    print(f"Threshold (Confidence): {conf_thres}")
    print(f"Total Defective Images Evaluated: {total_defective_images}")
    print("-" * 60)
    
    if total_defective_images > 0:
        m1_acc = (metric1_success / total_defective_images) * 100
        m2_acc = (metric2_success / total_defective_images) * 100
        
        print(f"Metric 1: YOLO + Secondary Label Only")
        print(f"  Condition: Model predicted at least one Defect Region (Class 2)")
        print(f"  Success:   {metric1_success} / {total_defective_images} -> [{m1_acc:.2f}% Accuracy]")
        print()
        print(f"Metric 2: YOLO + Primary + Secondary Labels")
        print(f"  Condition: Model predicted BOTH Defective Insulator (Class 1) AND Defect Region (Class 2)")
        print(f"  Success:   {metric2_success} / {total_defective_images} -> [{m2_acc:.2f}% Accuracy]")
        
        if save_failed:
            print("\n" + "-" * 60)
            print(f"Saved failed prediction visualizations to: {output_dir}")
            
        # Generate and save visualization
        plt.figure(figsize=(10, 6))
        metrics = ['YOLO + Secondary\n(Defect Region Only)', 'YOLO + Primary + Secondary\n(Both Insulator & Defect)']
        accuracies = [m1_acc, m2_acc]
        
        bars = plt.bar(metrics, accuracies, color=['#4CAF50', '#2196F3'])
        plt.ylim(0, 110) # 0 to 110% for head room
        plt.ylabel('Accuracy (%)')
        plt.title(f'Insulator Defect Detection Accuracy\n(Confidence Threshold: {conf_thres})')
        
        # Add text labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
            
        plot_path = 'accuracy_metrics_chart.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved visual chart to: {plot_path}")
        
    else:
        print("No defective images found in the dataset with Class 2.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./runs/detect/train_v2/weights/best.pt')
    parser.add_argument('--images-dir', type=str, default='./yolo_insulator_project/data_v2/images/test')
    parser.add_argument('--labels-dir', type=str, default='./yolo_insulator_project/data_v2/labels/test')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--save-failed', action='store_true', default=False, help="Save failed predictions")
    args = parser.parse_args()
    
    evaluate(args.weights, args.images_dir, args.labels_dir, args.conf, args.save_failed)
