from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(use_augmentation=True):
    transform_steps = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    ]

    if use_augmentation:
        transform_steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])

    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

    return transforms.Compose(transform_steps)


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def load_datasets(data_dir, use_augmentation=True):
    data_dir = Path(data_dir)

    train_dataset = datasets.ImageFolder(
        data_dir / "training",
        transform=get_train_transform(use_augmentation)
    )

    validation_dataset = datasets.ImageFolder(
        data_dir / "validation",
        transform=get_eval_transform()
    )

    evaluation_dataset = datasets.ImageFolder(
        data_dir / "evaluation",
        transform=get_eval_transform()
    )

    return train_dataset, validation_dataset, evaluation_dataset


def create_dataloaders(
    data_dir,
    batch_size=32,
    num_workers=0,
    use_augmentation=True
):
    train_dataset, validation_dataset, evaluation_dataset = load_datasets(
        data_dir=data_dir,
        use_augmentation=use_augmentation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, validation_loader, evaluation_loader