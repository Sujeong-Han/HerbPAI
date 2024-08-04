# HerbPAI: Herbarium image Preprocessing for AI Identification
- This is software designed to automate the preprocessing of plant specimen images. 
- It enables the detection and removal of non-plant specimen components from plant specimen image datasets.
<p align="center"><img src="https://github.com/user-attachments/assets/5799a4c5-1344-4d84-9b09-ae3877ad5965"></p>

  
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
![original](https://github.com/user-attachments/assets/070444f2-9349-4ec4-961f-7cfa49c6e825) |![Before2](https://github.com/user-attachments/assets/80fb7b21-6e0e-429a-b5b0-63d499fd1899)
--- | --- | 

## Ex) Data Label
![Example of labeling non-plant specimen components](https://github.com/user-attachments/assets/f1042557-69c4-4254-810b-d95e5b9fea90) |![process](https://github.com/user-attachments/assets/9479a1f3-7634-44b5-b1d6-781d54bfe0d2)
--- | --- | 

## Ex) Preprocessed Image
![After(digital)](https://github.com/user-attachments/assets/4acc2fb4-6de1-408f-b0db-75431591694b)|![after2](https://github.com/user-attachments/assets/47d77097-1add-47c3-a0b6-b9ffc8948f52)
--- | --- | 


