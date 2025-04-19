import torch
import numpy as np
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image

CLASS_LABELS = {
    0: "Background", 1: "Aeroplane", 2: "Bicycle", 3: "Bird", 4: "Boat",
    5: "Bottle", 6: "Bus", 7: "Car", 8: "Cat", 9: "Chair",
    10: "Cow", 11: "Dining Table", 12: "Dog", 13: "Horse", 14: "Motorbike",
    15: "Person", 16: "Potted Plant", 17: "Sheep", 18: "Sofa", 19: "Train", 20: "TV Monitor"
}


PALETTE = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]
], dtype=np.uint8)

def load_model():
    model = deeplabv3_resnet101(pretrained=True)
    model.eval()
    return model

def preprocess_image(image):
    # Convert PIL to OpenCV format
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

    # Apply median blur (noise removal)
    open_cv_image = cv2.medianBlur(open_cv_image, 3)

    # Convert back to PIL
    image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))

    transform = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def get_segmentation_mask(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    return output.argmax(0).byte().cpu().numpy()

def decode_segmentation(mask):
    return PALETTE[mask % len(PALETTE)]

def overlay_segmentation(image, mask_rgb, alpha=0.6):
    image = image.resize((mask_rgb.shape[1], mask_rgb.shape[0])).convert("RGBA")
    mask_img = Image.fromarray(mask_rgb).convert("RGBA")
    return Image.blend(image, mask_img, alpha)

def get_class_legend(mask):
    unique_ids = np.unique(mask)
    return [(CLASS_LABELS[i], PALETTE[i]) for i in unique_ids if i in CLASS_LABELS]

