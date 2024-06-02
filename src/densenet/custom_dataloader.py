import os
import random
import logging
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, verbosity=logging.INFO):
        """
        Args:
            root_dir (string): Directory with all the images divided into subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
            verbosity (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        self.root_dir = root_dir
        self.transform = transform
        logging.basicConfig(level=verbosity)

        self.subdirectories = [os.path.join(root_dir, o) for o in os.listdir(root_dir)
                       if os.path.isdir(os.path.join(root_dir, o))]

        logging.debug(f"Found subdirectories: {self.subdirectories}")

        self.images_per_class = {}
        for subdir in self.subdirectories:
            images = [os.path.join(subdir, img) for img in os.listdir(subdir)]
            self.images_per_class[subdir] = images

        self.anchor_images = [img for sublist in self.images_per_class.values() for img in sublist]
        logging.info(f"Total anchor images found: {len(self.anchor_images)}")

    def __len__(self):
        return len(self.anchor_images)

    def __getitem__(self, idx):
        anchor_img_path = self.anchor_images[idx]
        positive_img_path = self._get_positive_sample(anchor_img_path)
        negative_img_path = self._get_negative_sample(anchor_img_path)

        if not os.path.exists(anchor_img_path):
            raise FileNotFoundError(f"Anchor image path does not exist: {anchor_img_path}")
        if not os.path.exists(positive_img_path):
            raise FileNotFoundError(f"Positive image path does not exist: {positive_img_path}")
        if not os.path.exists(negative_img_path):
            raise FileNotFoundError(f"Negative image path does not exist: {negative_img_path}")

        try:
            anchor_img = Image.open(anchor_img_path).convert("RGB")
            positive_img = Image.open(positive_img_path).convert("RGB")
            negative_img = Image.open(negative_img_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error opening image: {e}")
            raise

        image_paths = [anchor_img_path, positive_img_path, negative_img_path]

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return (anchor_img, positive_img, negative_img), image_paths


    def _get_positive_sample(self, anchor_img_path):
        same_class_dir = os.path.dirname(anchor_img_path)
        all_images = self.images_per_class[same_class_dir]
        positive_img_path = anchor_img_path

        if len(all_images) > 1:
            while positive_img_path == anchor_img_path:
                positive_img_path = random.choice(all_images)
        logging.debug(f"Selected positive image path: {positive_img_path}")
        return positive_img_path

    def _get_negative_sample(self, anchor_img_path):
        """Returns a path to a negative sample, i.e., an image of a different individual."""
        different_class_dir = random.choice(self.subdirectories)
        # Ensure the selected directory is not the same as the anchor's and is not empty
        while os.path.dirname(anchor_img_path) == different_class_dir or not self.images_per_class[different_class_dir]:
            different_class_dir = random.choice(self.subdirectories)

        # Once a valid directory with images is found, choose an image from it
        all_images = self.images_per_class[different_class_dir]
        negative_img_path = os.path.join(different_class_dir, os.path.basename(random.choice(all_images)))

        # Ensure the path is absolute to prevent issues
        negative_img_path = os.path.abspath(negative_img_path)
        logging.debug(f"Constructed negative path: {negative_img_path}")  # Debugging output
        return negative_img_path