import os
from torchvision import datasets, transforms
from PIL import Image

def download_and_save_mnist_images(destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    print("Downloading MNIST dataset using torchvision...")
    train_dataset = datasets.MNIST(
        root=destination_folder,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    test_dataset = datasets.MNIST(
        root=destination_folder,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    # Save training images
    train_folder = os.path.join(destination_folder, "train")
    os.makedirs(train_folder, exist_ok=True)
    for idx, (image, label) in enumerate(train_dataset):
        label_folder = os.path.join(train_folder, str(label))
        os.makedirs(label_folder, exist_ok=True)
        image_path = os.path.join(label_folder, f"{idx}.png")
        image = transforms.ToPILImage()(image)
        image.save(image_path)

    # Save testing images
    test_folder = os.path.join(destination_folder, "test")
    os.makedirs(test_folder, exist_ok=True)
    for idx, (image, label) in enumerate(test_dataset):
        label_folder = os.path.join(test_folder, str(label))
        os.makedirs(label_folder, exist_ok=True)
        image_path = os.path.join(label_folder, f"{idx}.png")
        image = transforms.ToPILImage()(image)
        image.save(image_path)

    print(f"MNIST images saved to {destination_folder}.")

if __name__ == "__main__":
    destination = "data/mnist"
    download_and_save_mnist_images(destination)
