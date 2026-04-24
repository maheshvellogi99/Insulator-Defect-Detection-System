from ultralytics import YOLO

# Make sure to point to the correct train folder! (Try train2 if train doesn't work)
model = YOLO('./runs/detect/train2/weights/best.pt')

# We added conf=0.1 to force it to show lower-confidence guesses
results = model.predict(source='img6.jpg', save=True, show=True, conf=0.1)

print("Prediction complete! Look for the image popping up.")