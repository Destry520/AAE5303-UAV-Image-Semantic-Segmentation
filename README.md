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
 # Model
model = UNet(n_channels=3, n_classes=6)

 # Training Configuration
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
epochs = 10
batch_size = 2

## 5. Hyperparameter Tuning Results
4 compared experiments are finished.

| Group | LR | Optimizer | Final Loss | Accuracy | mIoU |
|-------|----|-----------|------------|----------|------|
| 1 | 1e-5 | ORIGIN | 0.5445 | 0.9421 | 0.6512 |
| 2 | 1e-5 | Adam | 0.3827 | 0.9426 | 0.6455 |
| 3 | 5e-5 | Adam | 0.2110 | 0.9291 | 0.6355 |
| 4 | 1e-5 | AdamW | 0.3631 | 0.9382 | 0.6377 |

**最优参数：第 2 组 (lr=1e-5 + Adam)**

## 6. Final Result
- Final Loss: 0.3827
- Accuracy: 0.9426
- mIoU: 0.6455

## 7. Result Show
<img width="693" height="232" alt="image" src="https://github.com/user-attachments/assets/d95d5ad6-c55e-4d5f-bf53-c049cfefe8f9" />


## 8. How to Run
Prepare dataset images and masks
Install dependencies
Run training script
Evaluate metrics (Acc, mIoU)
Visualize prediction results

## 9. Project Structure
├── train.py # training code

├── model.py # UNet model

├── final_model.pth # trained model

├── result.png # visualization

└── README.md

## 10. Contributions
- Responsible for the semantic segmentation module of the unmanned aerial vehicle images
- Completed the construction, training and debugging of the UNet model
