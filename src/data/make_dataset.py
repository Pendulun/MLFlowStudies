from torchvision.datasets import StanfordCars
from torchvision import transforms
from pathlib import Path

if __name__ == "__main__":
    DATA_FOLDER_PATH = Path("../../data/raw")

    if not DATA_FOLDER_PATH.exists():
        DATA_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    
    train_dataset = StanfordCars(root=DATA_FOLDER_PATH,
                             split='train',
                             transform=transforms.ToTensor(),
                             target_transform=None,
                             download=True
                             )

    test_dataset = StanfordCars(root=DATA_FOLDER_PATH,
                             split='test',
                             transform=transforms.ToTensor(),
                             target_transform=None,
                             download=True
                             )
    
    print(f"train_dataset len: {len(train_dataset)}")
    print(f"test_dataset len: {len(test_dataset)}")