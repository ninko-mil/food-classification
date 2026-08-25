EXPERIMENT_CONFIGS = [
    {
        "name": "exp_01_resnet_no_aug",
        "model": "resnet18",
        "learning_rate": 0.001,
        "optimizer": "adam",
        "augmentation": False,
        "batch_size": 64
    },
    {
        "name": "exp_02_resnet_aug",
        "model": "resnet18",
        "learning_rate": 0.001,
        "optimizer": "adam",
        "augmentation": True,
        "batch_size": 64
    },
    {
        "name": "exp_03_resnet_aug_lr_0005",
        "model": "resnet18",
        "learning_rate": 0.0005,
        "optimizer": "adam",
        "augmentation": True,
        "batch_size": 64
    },
    {
        "name": "exp_04_resnet_aug_adamw",
        "model": "resnet18",
        "learning_rate": 0.0001,
        "optimizer": "adamw",
        "augmentation": True,
        "batch_size": 64
    },
    {
        "name": "exp_05_resnet_aug_sgd",
        "model": "resnet18",
        "learning_rate": 0.001,
        "optimizer": "sgd",
        "augmentation": True,
        "batch_size": 64,
        "momentum": 0.9
    }
]