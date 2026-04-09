# AAE5303 UAV Image Semantic Segmentation
Group Project - Semantic Segmentation Part

## 1. Task Description
Responsible for semantic segmentation on UAV aerial images.
Build and train a UNet model to classify each pixel into 6 categories.

## 2. Model Architecture
- Model: UNet
- Input: 3-channel RGB image (256x256)
- Output: 6-class segmentation map
- Structure: Encoder + Decoder + Skip Connections

## 3. Dataset
- Dataset: UAVScenes (MARS-LVIG)
- Total images: 1380
- Image size: 256x256
- Number of classes: 6

## 4. Class Definitions
0: Background
1: Road
2: Building
3: Vegetation
4: Tree
5: Car

## 5. Environment
- Python 3.x
- PyTorch
- torchvision
- numpy
- tqdm
- torchmetrics
- Pillow

## 6. Implementation Code
### UNet Model
```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
```
# Training Code
```
model = UNet(n_channels=3, n_classes=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")
```
# Evaluation Code
```
from torchmetrics import JaccardIndex, Accuracy

miou = JaccardIndex(task="multiclass", num_classes=6).to(device)
acc = Accuracy(task="multiclass", num_classes=6).to(device)

model.eval()
with torch.no_grad():
    for img, mask in dataloader:
        img, mask = img.to(device), mask.to(device)
        pred = model(img).argmax(dim=1)
        miou.update(pred, mask)
        acc.update(pred, mask)

final_acc = acc.compute().item()
final_miou = miou.compute().item()
```
# Visualization Code
```
import matplotlib.pyplot as plt

img, mask = dataset[100]
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device)).argmax(1).squeeze().cpu().numpy()

plt.figure(figsize=(12,4))
plt.subplot(131); plt.imshow(img.permute(1,2,0))
plt.subplot(132); plt.imshow(mask)
plt.subplot(133); plt.imshow(pred)
plt.show()
```
## 7. Hyperparameter Tuning Results
| Group | LR | Optimizer | Final Loss | Accuracy | mIoU |
|-------|----|-----------|------------|----------|------|
| 1 | 1e-5 | ORIGIN | 0.5445 | 0.9421 | 0.6512 |
| 2 | 1e-5 | Adam | 0.3827 | 0.9426 | 0.6455 |
| 3 | 5e-5 | Adam | 0.2110 | 0.9291 | 0.6355 |
| 4 | 1e-5 | AdamW | 0.3631 | 0.9382 | 0.6377 |
## 8. Final Result
Final Loss: 0.3827
Accuracy: 0.9426
mIoU: 0.6455
## 9. Result Show
<img width="693" height="232" alt="image" src="https://github.com/user-attachments/assets/d95d5ad6-c55e-4d5f-bf53-c049cfefe8f9" />


## 10. How to Run
Install required packages
Prepare dataset
Run training script
Evaluate metrics
Visualize results

## 11. Project Structure
├── train.py # training code

├── model.py # UNet model

├── final_model.pth # trained model

├── result.png # visualization

└── README.md

## 12. Contributions
Responsible for UAV image semantic segmentation
Built and trained UNet model
Completed hyperparameter tuning
Evaluated model performance
Visualized segmentation results
