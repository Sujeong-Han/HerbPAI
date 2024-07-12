# Automated Preprocess Herbarium SW
- This is software designed to automate the preprocessing of plant specimen data. 
- It enables the detection and removal of non-plant specimen components from plant specimen image datasets.

## Ex) Data Label
![Example of labeling non-plant specimen components](https://github.com/user-attachments/assets/f1042557-69c4-4254-810b-d95e5b9fea90)

# Modify detect_dual.py

- Detect all objects except plant specimens in an image using the trained YOLO v9-e model.
- After that, fill the bounding box of detected objects with white.
- This also detects the edges of the sample data and has been modified to remove outliers.

# Inference
 ```
python detect_dual.py --source example.jpg' --img 640 --device 0 --weights 'best.pt' --name test
 ```

- You can find model weight in google drive.
- If you run above inference code, you can find output image in test folder. 





