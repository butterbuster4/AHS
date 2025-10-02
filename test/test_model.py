'''
import torch
from torchvision import transforms
from PIL import Image

# 1. transformer
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 2. load the model
from torchvision import models
model = models.efficientnet_b3(pretrained=False)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, 2)

# load weights
model.load_state_dict(torch.load("./models/best_efficientnetb3.pth", map_location="cpu"))
model.eval()

# 3. single image
img_path = "best_individual.jpg"
img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0)  # [1, 3, 300, 300]

# 4. predict
with torch.no_grad():
    outputs = model(img)
    probs = torch.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

print("预测类别:", pred_class)   # 1 = assembled, 0 = not assembled
print("概率分布:", probs.numpy())
'''

import torch
from torchvision import transforms
from PIL import Image
from torchvision import models

# 1. transformer
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 2. load the model
model = models.efficientnet_b3(pretrained=False)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, 2)

# load weights
model.load_state_dict(torch.load("./models/best_efficientnetb3.pth", map_location="cpu"))
model.eval()

# 3. single image
img_path = "done.jpg"
img = Image.open(img_path).convert("RGB")
img = transform(img).unsqueeze(0)  # [1, 3, 300, 300]

# 4. predict
with torch.no_grad():
    outputs = model(img)
    probs = torch.softmax(outputs, dim=1)  # [batch, 2]
    assembled_prob = probs[0][1].item()    # assembled 的概率
    not_assembled_prob = probs[0][0].item()
    pred_class = torch.argmax(probs, dim=1).item()

print(f"预测类别: {pred_class} (0 = not assembled, 1 = assembled)")
print(f"拼好(assembled)的可能性: {assembled_prob:.4f}")
print(f"没拼好(not assembled)的可能性: {not_assembled_prob:.4f}")
