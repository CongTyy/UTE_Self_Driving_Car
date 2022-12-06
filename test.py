import cv2
import numpy as np
import torch
import time
from model import build_unet

checkpoint_path = "output_1.pth"
model = build_unet().cuda()
model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
model.eval()

# 159, 396
img = cv2.imread("noise_lane.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (160, 80))

x = torch.from_numpy(img).cuda()
x = x.transpose(1, 2).transpose(0, 1)
x = x / 255.0
x = x.unsqueeze(0).float()
print(x.size())
with torch.no_grad():
    pred = model(x)
    pred = torch.sigmoid(pred)
    pred = pred[0].squeeze()
    pred = (pred > 0.1).cpu().numpy()

    pred = np.array(pred, dtype=np.uint8)
    pred = pred * 255

    cv2.imwrite("pred.png", cv2.resize(pred, (396, 159)))