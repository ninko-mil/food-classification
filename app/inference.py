from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from torchvision import transforms

from src.model import create_resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_food11_resnet18.pth"

CLASS_NAMES = [
    "bread",
    "dairy_product",
    "dessert",
    "egg",
    "fried_food",
    "meat",
    "noodles_pasta",
    "rice",
    "seafood",
    "soup",
    "vegetable_fruit",
]

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


inference_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


class FoodClassifier:
    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = create_resnet18(
            num_classes=len(CLASS_NAMES)
        )

        state_dict = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image, top_k: int = 3):
        image = image.convert("RGB")

        tensor = inference_transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        start_time = perf_counter()

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_time_ms = (
            perf_counter() - start_time
        ) * 1000

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=min(top_k, len(CLASS_NAMES)),
            dim=1,
        )

        predictions = []

        for probability, index in zip(
            top_probabilities[0],
            top_indices[0],
        ):
            predictions.append(
                {
                    "class": CLASS_NAMES[index.item()],
                    "probability": probability.item(),
                }
            )

        return {
            "predicted_class": predictions[0]["class"],
            "confidence": predictions[0]["probability"],
            "top_predictions": predictions,
            "inference_time_ms": inference_time_ms,
            "device": str(self.device),
        }


if __name__ == "__main__":
    print("Inference module loaded successfully.")
    print(f"Model path: {MODEL_PATH}")