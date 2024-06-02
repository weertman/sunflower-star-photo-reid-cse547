from torchvision.transforms import functional as F
from PIL import Image
import random
from torchvision import transforms

class RandomPad(object):
    def __init__(self, max_pad, fill=0, padding_mode='constant'):
        assert isinstance(max_pad, (int, tuple))
        if isinstance(max_pad, int):
            self.max_pad = (max_pad, max_pad)
        else:
            assert len(max_pad) == 2, "max_pad should be int or 2-tuple"
            self.max_pad = max_pad

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        padding = tuple(random.randint(0, max_pad) for max_pad in self.max_pad)
        return F.pad(img, padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(max_pad={0}, fill={1}, padding_mode={2})'.\
            format(self.max_pad, self.fill, self.padding_mode)

class RandomGaussianBlur(object):
    def __init__(self, kernel_size_range, sigma_range=(0.1, 2.0), p=0.5):
        assert isinstance(kernel_size_range, tuple) and len(kernel_size_range) == 2, \
            "kernel_size_range should be a tuple (min_size, max_size)"
        self.kernel_size_range = kernel_size_range
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            # Select a random kernel size within the range
            kernel_size = random.randint(self.kernel_size_range[0], self.kernel_size_range[1])
            # Ensure the kernel size is always an odd number (required for the Gaussian blur operation)
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
            sigma = random.uniform(*self.sigma_range)
            return transforms.GaussianBlur(kernel_size, sigma)(img)
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(kernel_size_range={self.kernel_size_range}, sigma_range={self.sigma_range}, p={self.p})"

class RandomColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.5):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(brightness={0}, contrast={1}, saturation={2}, hue={3}, p={4})'.\
            format(self.transform.brightness, self.transform.contrast,
                   self.transform.saturation, self.transform.hue, self.p)

class RandomProportionalCrop(object):
    def __init__(self, scale_range=(0.8, 1.0), aspect_ratio_range=(3/4, 4/3)):
        self.scale_range = scale_range
        self.aspect_ratio_range = aspect_ratio_range

    def __call__(self, img):
        scale = (self.scale_range[0], self.scale_range[1])
        ratio = (self.aspect_ratio_range[0], self.aspect_ratio_range[1])
        return transforms.RandomResizedCrop(size=img.size, scale=scale, ratio=ratio)(img)

    def __repr__(self):
        return f"{self.__class__.__name__}(scale_range={self.scale_range}, aspect_ratio_range={self.aspect_ratio_range})"

class RandomOrderTransforms(object):
    def __init__(self, transforms, probs=None, list_prob=1.0, max_n_transforms=4):
        self.transforms = transforms
        if probs is None:
            self.probs = [1.0] * len(transforms)
        else:
            self.probs = probs
        self.list_prob = list_prob
        self.max_n_transforms = max_n_transforms

    def __call__(self, img):
        if random.random() < self.list_prob:
            order = list(range(len(self.transforms)))
            random.shuffle(order)
            count_transforms = 0
            for i in order:
                if random.random() < self.probs[i]:
                    img = self.transforms[i](img)
                    count_transforms += 1
                    if count_transforms >= self.max_n_transforms:
                        break
        return img


def get_transforms(h, w,
                   max_pad=30,
                   rotation_degrees=90,
                   affine_degrees=20, affine_translate=(0.2, 0.2),
                   perspective_distortion=0.25,
                   blur_kernel_range=(3, 5), blur_sigma_range=(0.1, 1),
                   jitter_params={'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.2},
                   crop_scale_range=(0.5, 1), crop_aspect_ratio_range=(3 / 4, 4 / 3),
                   interpolation=3, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                   transform_probs=None, list_prob=1):
    # Define valid keys for transform probabilities
    valid_transforms = {
        'RandomPad', 'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomRotation',
        'RandomAffine', 'RandomPerspective', 'RandomGaussianBlur', 'RandomColorJitter',
        'RandomProportionalCrop'
    }

    # Set default probabilities if none provided
    if transform_probs is None:
        transform_probs = {key: 1 for key in valid_transforms}
    else:
        # Check for any invalid keys in the provided dictionary
        invalid_keys = set(transform_probs.keys()) - valid_transforms
        if invalid_keys:
            raise ValueError(f"Invalid keys in transform_probs: {invalid_keys}. Valid keys are: {valid_transforms}")

    random_order_transforms = [
        RandomPad(max_pad=max_pad),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(rotation_degrees),
        transforms.RandomAffine(degrees=affine_degrees, translate=affine_translate),
        transforms.RandomPerspective(distortion_scale=perspective_distortion, p=1),
        RandomGaussianBlur(kernel_size_range=blur_kernel_range, sigma_range=blur_sigma_range, p=1),
        RandomColorJitter(brightness=jitter_params['brightness'], contrast=jitter_params['contrast'],
                          saturation=jitter_params['saturation'], hue=jitter_params['hue'], p=1),
        RandomProportionalCrop(scale_range=crop_scale_range, aspect_ratio_range=crop_aspect_ratio_range)
    ]

    # Build the transform probabilities list based on the provided dictionary
    probability_list = [transform_probs.get(key, 1) for key in valid_transforms]

    always_end_transforms = [
        transforms.Resize((h, w), interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    transform_train_list = [
                               RandomOrderTransforms(random_order_transforms, probability_list, list_prob),
                           ] + always_end_transforms

    transform_test_list = [
                              RandomOrderTransforms(random_order_transforms, probability_list, list_prob),
                          ] + always_end_transforms

    return transform_train_list, transform_test_list


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    h, w = 256, 256

    transform_probs = {
        'RandomPad': 0.9,  # 90% chance to apply RandomPad
        'RandomHorizontalFlip': 0.5,  # 50% chance to apply RandomHorizontalFlip
        'RandomVerticalFlip': 0.5,  # 50% chance to apply RandomVerticalFlip
        'RandomRotation': 0.7,  # 70% chance to apply RandomRotation
        'RandomAffine': 0.6,  # 60% chance to apply RandomAffine
        'RandomPerspective': 0.4,  # 40% chance to apply RandomPerspective
        'RandomGaussianBlur': 0.8,  # 80% chance to apply RandomGaussianBlur
        'RandomColorJitter': 0.5,  # 50% chance to apply RandomColorJitter
        'RandomProportionalCrop': 0.7  # 70% chance to apply RandomProportionalCrop
    }

    transform_train_list, transform_test_list = get_transforms(h, w, transform_probs=transform_probs)
    print(transform_train_list)
    print(transform_test_list)

    path_img = 'transforms_test.png'
    img = Image.open(path_img).convert("RGB")
    width = img.size[0]
    height = img.size[1]

    n_examples = 3
    for k in range(n_examples):
        nrows = 4
        ncols = 2
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(int(ncols * 2), int(nrows * 2)) if nrows > 1 else (5, 5))
        for i in range(nrows):
            start_time = time.time()
            transform_train = transforms.Compose(transform_train_list)
            img_transformed = transform_train(img)
            print(f"Transformed in {time.time() - start_time:.2f} seconds")

            ax = axs[i, 0]
            ax.imshow(img)
            yticks = [0, height]
            xticks = [0, width]

            ax.set_yticks(yticks)
            if i == 0:
                ax.set_title("Original")

            if i == nrows-1:
                ax.set_xticks(xticks)
            else:
                ax.set_xticks([])

            img_transformed = img_transformed.permute(1, 2, 0)
            ## normalize
            img_transformed = (img_transformed - img_transformed.min()) / (img_transformed.max() - img_transformed.min())

            yticks = [0, img_transformed.shape[0]]
            xticks = [0, img_transformed.shape[1]]

            ax = axs[i, 1]
            ax.imshow(img_transformed)

            ## apply yticks to the opposite side of subplot than is normal
            ax.set_yticks(yticks)
            ax.yaxis.tick_right()

            if i == nrows-1:
                ax.set_xticks(xticks)
            else:
                ax.set_xticks([])

            if i == 0:
                ax.set_title("Transformed")

        plt.show()
        plt.close()

