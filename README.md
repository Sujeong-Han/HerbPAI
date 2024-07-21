# Automated Preprocess Herbarium SW
- This is software designed to automate the preprocessing of plant specimen images. 
- It enables the detection and removal of non-plant specimen components from plant specimen image datasets.

# Modify detect_dual.py

- Detect all objects except plant in an image using the trained YOLO v9-e model.
- After that, fill the bounding box of detected objects with white.
- Additionally, for images taken with digital cameras, unnecessary information outside the specimen.
  
# Inference
 ```
python detect_dual.py --source example.jpg' --img 640 --device 0 --weights 'best.pt' --name test
 ```

- You can find model weight in google drive.
- If you run above inference code, you can find output image in test folder. 

## Ex) Original Image
![Before](https://github.com/user-attachments/assets/e5b840ec-03db-4f23-9083-6db1f532ccf0) |![Before2](https://github.com/user-attachments/assets/80fb7b21-6e0e-429a-b5b0-63d499fd1899)
--- | --- | 

## Ex) Data Label
![Example of labeling non-plant specimen components](https://github.com/user-attachments/assets/f1042557-69c4-4254-810b-d95e5b9fea90) |![process](https://github.com/user-attachments/assets/9479a1f3-7634-44b5-b1d6-781d54bfe0d2)
--- | --- | 

## Ex) Preprocessed Image
![After](https://github.com/user-attachments/assets/4a17445c-8c13-4a9c-aff0-40d7666ba5d4)|![After2](https://github.com/user-attachments/assets/50bb97cc-f545-49de-aea8-94b350c2c4ea)
--- | --- | 


