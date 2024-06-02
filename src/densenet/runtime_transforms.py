import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


class RuntimeTransforms:
    def __init__(self, h, w, resize_ratio=0.5):
        self.h = h
        self.w = w

        self.resize_h = int(self.h * resize_ratio)
        self.resize_w = int(self.w * resize_ratio)
        self.transforms_list = [
            transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
            transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
            transforms.Compose([]),
            transforms.Compose([transforms.Resize((self.h, self.w)),
                                transforms.CenterCrop((self.resize_h, self.resize_w))]),  # No change
            transforms.Compose([transforms.RandomVerticalFlip(p=1.0),
                                transforms.CenterCrop((self.resize_h, self.resize_w))]),
            transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                                transforms.CenterCrop((self.resize_h, self.resize_w))]),
        ]
        self.always_end_transforms = transforms.Compose([
            transforms.Resize((self.h, self.w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def apply_transforms(self, img):
        transformed_images = []
        for transform in self.transforms_list:
            transformed_img = transform(img)
            transformed_img = self.always_end_transforms(transformed_img)
            transformed_images.append(transformed_img)
        return transformed_images


# Function to denormalize the image for visualization
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# Example usage
if __name__ == "__main__":
    img_path = 'transforms_test.png'
    img = Image.open(img_path).convert("RGB")

    # Assuming original dimensions of the image
    original_h, original_w = img.size

    runtime_transforms = RuntimeTransforms(original_h, original_w)
    transformed_images = runtime_transforms.apply_transforms(img)

    # Plot original and transformed images
    fig, axs = plt.subplots(1, len(transformed_images) + 1, figsize=(15, 5))

    # Plot original image
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Mean and std for denormalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Plot transformed images
    for idx, transformed_img in enumerate(transformed_images):
        denorm_img = denormalize(transformed_img.clone(), mean, std)
        transformed_img_pil = transforms.ToPILImage()(denorm_img)
        axs[idx + 1].imshow(transformed_img_pil)
        axs[idx + 1].set_title(f'Transformed {idx + 1}')
        axs[idx + 1].axis('off')

    plt.savefig('runtime_transforms.png',dpi=300)

    plt.show()

