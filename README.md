# AAE5303-UAV-Image-Semantic-Segmentation
Group Project - Semantic Segmentation Part

## 1. Task Introduction
Responsible for semantic segmentation on UAV aerial images.
Build and train a UNet model to classify each pixel into 6 categories.

## 2. Model
- Model: UNet
- Input: 3-channel RGB image (256x256)
- Output: 6-class semantic segmentation map
- Structure: Encoder + Decoder + Skip Connections

---
## 3. Data
- Dataset: UAVScenes (MARS-LVIG)
- Number of images: 1380
- Image size: 256x256
- Number of categories: 6

## 4. Implementation Code (Core Part)
```python
# Model
model = UNet(n_channels=3, n_classes=6)

# Training Configuration
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
epochs = 10
batch_size = 2

# Training Loop
for epoch in range(epochs):
    model.train()
    for img, mask in dataloader:
        pred = model(img)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()


- Framework: PyTorch
- Device: Kaggle T4 GPU
- Optimizer: Adam
- Learning rate: 1e-5
- Loss function: CrossEntropyLoss
- Epoch: 10
- 
## 5. Hyperparameter Tuning Results

4 compared experiments are finished

| Group | LR | Optimizer | Final Loss | Accuracy | mIoU |
|1  |1e-5 |ORIGIN|  0.5445 |0.9421|0.6512|
| 1 | 1e-5 | Adam | 0.3827 | 0.9426 | 0.6455 |
| 2 | 5e-5 | Adam | 0.2110 | 0.9291 | 0.6355 |
| 3 | 1e-5 | AdamW | 0.3631 | 0.9382 | 0.6377 |
<img width="449" height="300" alt="image" src="https://github.com/user-attachments/assets/8d6f7c0c-ca61-41c5-bfdd-31cf83c729ce" />
<img width="526" height="335" alt="image" src="https://github.com/user-attachments/assets/92973051-abc5-4335-a85a-f1b9edb8e9ab" />
<img width="402" height="259" alt="image" src="https://github.com/user-attachments/assets/594a3fb2-abc0-4869-a46a-4113bd7dd3b0" />
<img width="406" height="283" alt="image" src="https://github.com/user-attachments/assets/89266fa4-0a47-406b-93e1-984b05e77088" />


**最优参数：第 2 组（lr=1e-5 + Adam）**

## 6. Final result
- Final Loss: 0.3827
- Accuracy: 0.9426
- mIoU: 0.6455
  
## 7. Result show

<img width="693" height="232" alt="image" src="https://github.com/user-attachments/assets/6ad07e61-e420-473a-ba12-07df457a9f3d" />

## 8. How to Run
Prepare dataset images and masks
Install dependencies
Run training script
Evaluate metrics (Acc, mIoU)
Visualize prediction results

## 9. Project Structure
├── train.py            # training code
├── model.py            # UNet model
├── final_model.pth     # trained model
├── result.png          # visualization
└── README.md
## 10. Contributions
- Responsible for the semantic segmentation module of the unmanned aerial vehicle images
- Completed the construction, training and debugging of the UNet model
- Conducted comparative experiments on hyperparameters
- Carried out model evaluation and result visualization
