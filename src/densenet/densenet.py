import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights, DenseNet161_Weights, DenseNet169_Weights, DenseNet201_Weights
import torch

class CustomDenseNet(nn.Module):
    def __init__(self, version='densenet121', num_features=1024, embedding_dim=128, pretrained=True, num_embed_layers=2):
        super(CustomDenseNet, self).__init__()
        self.version = version
        self.model, num_features = self._get_densenet_model(version, pretrained)

        fanout = int(embedding_dim * 4)
        if fanout > num_features:
            print(f"Warning: fanout ({fanout}) is larger than the number of features ({num_features}). ")
            fanout = num_features

        # Remove the original classifier
        self.model.classifier = nn.Identity()

        # New embedding module with multiple layers, hourglass shape, batch normalization, and Xavier initialization
        if num_embed_layers == 1:
            self.embedding = nn.Sequential(
                nn.Linear(num_features, embedding_dim),
                nn.GELU()
            )
        elif num_embed_layers == 2:
            self.embedding = nn.Sequential(
                nn.Linear(num_features, int(embedding_dim*2)),
                nn.GELU(),
                nn.Linear(int(embedding_dim*2), embedding_dim),
                nn.GELU())
        elif num_embed_layers == 3:
            self.embedding = nn.Sequential(
                nn.Linear(num_features, fanout),
                nn.GELU(),
                nn.Linear(fanout, int(embedding_dim*2)),
                nn.GELU(),
                nn.Linear(int(embedding_dim*2), embedding_dim),
                nn.GELU())
        else:
            raise ValueError("You done fucked up")

        # Apply Xavier initialization to the linear layers in the embedding module
        self._init_weights()

        # Initially freeze all the pre-trained layers if pretrained is True
        if pretrained:
            self.freeze_pretrained()

    def count_layers(self):
        layer_counts = {}

        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                layer_type = 'Conv2d'
            elif isinstance(module, nn.Linear):
                layer_type = 'Linear'
            elif isinstance(module, nn.BatchNorm2d):
                layer_type = 'BatchNorm2d'
            elif isinstance(module, nn.ReLU):
                layer_type = 'ReLU'
            elif isinstance(module, nn.MaxPool2d):
                layer_type = 'MaxPool2d'
            elif isinstance(module, nn.AvgPool2d):
                layer_type = 'AvgPool2d'
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                layer_type = 'AdaptiveAvgPool2d'
            else:
                layer_type = type(module).__name__

            if layer_type not in layer_counts:
                layer_counts[layer_type] = 0
            layer_counts[layer_type] += 1

        total_layers = sum(layer_counts.values())

        #print("Layer counts:")
        #for layer_type, count in layer_counts.items():
        #    print(f"{layer_type}: {count}")
        #print(f"Total number of layers: {total_layers}")
    def _get_densenet_model(self, version, pretrained):
        models_dict = {
            'densenet121': (models.densenet121, DenseNet121_Weights.IMAGENET1K_V1, 1024),
            'densenet161': (models.densenet161, DenseNet161_Weights.IMAGENET1K_V1, 2208),
            'densenet169': (models.densenet169, DenseNet169_Weights.IMAGENET1K_V1, 1664),
            'densenet201': (models.densenet201, DenseNet201_Weights.IMAGENET1K_V1, 1920),
        }
        if version in models_dict:
            model_fn, weights, num_features = models_dict[version]
            if pretrained:
                model = model_fn(weights=weights)
            else:
                model = model_fn(weights=None)  # Load the model without pretrained weights
            return model, num_features
        else:
            raise ValueError(f"Unsupported DenseNet version: {version}. Supported versions are {list(models_dict.keys())}")

    def _init_weights(self):
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        return x

    def freeze_pretrained(self):
        # Freeze all parameters in the pre-trained model
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_n_layers(self, n):
        # Unfreeze the last n layers of the pretrained backbone
        layers = list(self.model.features.children())

        layer_counts = {}
        total_layers = len(layers)
        unfrozen_layers = layers[-n:]

        print(f"Unfreezing the last {n} layers out of {total_layers} total layers:")

        for layer in unfrozen_layers:
            layer_type = type(layer).__name__
            if layer_type not in layer_counts:
                layer_counts[layer_type] = 0
            layer_counts[layer_type] += 1

            for param in layer.parameters():
                param.requires_grad = True

        #print("Layer counts of unfrozen layers:")
        #for layer_type, count in layer_counts.items():
        #    print(f"{layer_type}: {count}")

        print(f"Unfrozen layers: {unfrozen_layers}")

    def unfreeze_all(self):
        # Unfreeze all parameters in the model for full training
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.embedding.parameters():
            param.requires_grad = True


from torch.utils.data import Dataset
import numpy as np
import random

class TripletMNISTDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.labels = np.array(mnist_dataset.targets, dtype=int)
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in np.unique(self.labels)}

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, index):
        img1, label1 = self.mnist_dataset[index]
        label1 = int(label1)

        positive_index = random.choice(self.label_to_indices[label1])
        while positive_index == index:  # Ensure positive is not the anchor
            positive_index = random.choice(self.label_to_indices[label1])

        negative_label = random.choice(list(set(self.labels) - {label1}))
        negative_index = random.choice(self.label_to_indices[negative_label])

        img2, _ = self.mnist_dataset[positive_index]
        img3, _ = self.mnist_dataset[negative_index]

        return (img1, img2, img3), (label1, label1, self.labels[negative_index])


if __name__ == '__main__':
    import numpy as np
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from tqdm import tqdm
    import torch.optim as optim
    import os
    from torch.optim.lr_scheduler import StepLR
    import datetime
    import pandas as pd
    from src.densenet.logging_visualizations import visualize_learning_logs

    def create_triplet_mnist_loader(mnist_dataset, batch_size, shuffle=True, num_workers=4, prefetch_factor=2):
        triplet_dataset = TripletMNISTDataset(mnist_dataset)
        return DataLoader(triplet_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, prefetch_factor=prefetch_factor)

    def dataloader_example(dataloader, fig_dir, fig_name, batch_size, plot_to=4):
        image_titles = ['Anchor', 'Positive', 'Negative']
        if plot_to > batch_size:
            plot_to = batch_size

        print('Example of a batch of images from the dataloader')
        for batch_idx, batch in enumerate(dataloader):
            print(f'len(batch): {len(batch)}')
            print(f'len(batch[0]): {len(batch[0])}')
            print(f'len(batch[1]): {len(batch[1])}')
            (anchor, positive, negative), labels = batch
            print(anchor.shape, positive.shape, negative.shape)

            fig, axs = plt.subplots(plot_to, 3, figsize=(6, 2 * plot_to))

            for i in range(plot_to):
                images = [anchor[i], positive[i], negative[i]]
                images = [img.permute(1, 2, 0) for img in images]
                images = [img.cpu().numpy() for img in images]
                images = [(img - img.min()) / (img.max() - img.min()) for img in images]

                for j in range(3):
                    ax = axs[i][j]
                    ax.imshow(images[j], cmap='gray')
                    if i == 0:
                        ax.set_title(image_titles[j])
                    label = labels[j][i].item()
                    ax.set_xlabel(f'{label}')

                    ax.set_xticks([])
                    ax.set_yticks([])

            fig.tight_layout()

            path_fig = os.path.join(fig_dir, f'{fig_name}.png')
            plt.savefig(path_fig, dpi=300)

            plt.show()
            plt.close()
            print(f"Batch {batch_idx + 1} out of {len(dataloader)}")
            break

    print(f'Torch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')

    print("Loading the MNIST dataset...")
    # Load the MNIST dataset
    from src.densenet.custom_transforms import get_transforms
    # Define the new parameters for calling get_transforms
    h = 256  # height for resizing images
    w = 256  # width for resizing images
    max_pad = 5  # max padding
    rotation_degrees = 45  # rotation
    affine_degrees = 15  # affine transformation rotation
    affine_translate = (0.15, 0.15)  # translation in affine transformations
    perspective_distortion = 0.3  # perspective distortion
    blur_kernel_range = (3, 5)  # blur kernel size range
    blur_sigma_range = (0.5, 2)  # sigma range for Gaussian blur
    jitter_params = {'brightness': 0.3, 'contrast': 0.3, 'saturation': 0.3, 'hue': 0.3}  # more drastic color changes
    crop_scale_range = (0.6, 0.9)  # narrower cropping scale range
    crop_aspect_ratio_range = (0.75, 1.25)  # slightly more varied aspect ratio range
    interpolation = 3
    mean = [0.5, 0.5, 0.5]  # changed mean normalization values
    std = [0.25, 0.25, 0.25]  # changed std deviation normalization values

    # Custom probability for each transform
    transform_probs = {
        'RandomPad': 0.8,
        'RandomHorizontalFlip': 0.6,
        'RandomVerticalFlip': 0.4,
        'RandomRotation': 0.5,
        'RandomAffine': 0.7,
        'RandomPerspective': 0.5,
        'RandomGaussianBlur': 0.9,
        'RandomColorJitter': 0.75,
        'RandomProportionalCrop': 0.85
    }

    # Probability of applying the list of transforms
    list_prob = 0.95  # 90% chance to apply the list of random transforms

    transform_train_list, transform_test_list = get_transforms(
        h, w,
        max_pad=max_pad,
        rotation_degrees=rotation_degrees,
        affine_degrees=affine_degrees, affine_translate=affine_translate,
        perspective_distortion=perspective_distortion,
        blur_kernel_range=blur_kernel_range, blur_sigma_range=blur_sigma_range,
        jitter_params=jitter_params,
        crop_scale_range=crop_scale_range, crop_aspect_ratio_range=crop_aspect_ratio_range,
        interpolation=interpolation, mean=mean, std=std,
        transform_probs=transform_probs, list_prob=list_prob
    )

    # Insert Grayscale conversion as the second step in the transform list for both training and testing
    transform_train_list.insert(1, transforms.Grayscale(num_output_channels=3))
    transform_test_list.insert(1, transforms.Grayscale(num_output_channels=3))

    transform_train_list = transforms.Compose(transform_train_list)
    transform_test_list = transforms.Compose(transform_test_list)
    print(f'Train transforms: {transform_train_list}')
    print(f'Test transforms: {transform_test_list}')

    mnist_directory = os.path.join('..', '..', 'data', 'mnist')
    train_dataset = datasets.MNIST(root=mnist_directory, train=True, download=True,
                                   transform=transform_train_list)
    test_dataset = datasets.MNIST(root=mnist_directory, train=False, download=True,
                                  transform=transform_test_list)

    # Create data loaders with triplets
    print("Creating data loaders...")
    batch_size = 16
    num_workers = 4
    prefetch_factor = 2
    train_loader = create_triplet_mnist_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               prefetch_factor=prefetch_factor)
    test_loader = create_triplet_mnist_loader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                              prefetch_factor=prefetch_factor)

    print("Creating the model...")
    # Instantiate the model and loss function
    pretrained = True
    model_version = 'densenet121'

    # Define the number of epochs
    num_epochs = 10
    unfreeze_n_epoch = 2
    unfreeze_all_epoch = 5
    n_unfreeze = 128

    model = CustomDenseNet(embedding_dim=128, version=model_version, pretrained=pretrained)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Define the triplet loss function
    margin = 0.5
    # Define the optimizer
    lr = 0.001
    # Define the scheduler
    gamma = 0.5
    step_size = 1
    # Define the number of epochs
    num_epochs = 50
    unfreeze_n_epoch = 10
    unfreeze_all_epoch = 25
    n_unfreeze = 128
    triplet_loss = nn.TripletMarginLoss(margin=margin)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    model_root_dir = os.path.join('..', '..', 'models', 'mnist_triplet_densenet')
    if not os.path.exists(model_root_dir):
        print(f'Creating directory: {model_root_dir}')
        os.makedirs(model_root_dir)

    date_txt_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    model_name = f'{model_version}__PT-{pretrained}.pth'
    model_dir = os.path.join(model_root_dir, date_txt_now + '__' + model_name.split('.')[0])
    if not os.path.exists(model_dir):
        print(f'Creating directory: {model_dir}')
        os.makedirs(model_dir)
    path_model = os.path.join(model_dir, model_name)
    path_best_model = os.path.join(model_dir, 'best_' + model_name)
    if not os.path.exists(path_model):
        torch.save(model.state_dict(), path_model)
    if not os.path.exists(path_best_model):
        torch.save(model.state_dict(), path_best_model)

    path_fig_dir = os.path.join(model_dir, 'figures')
    if not os.path.exists(path_fig_dir):
        os.makedirs(path_fig_dir)

    fig_name = 'train_batch_example'
    dataloader_example(train_loader, path_fig_dir, fig_name, batch_size=batch_size, plot_to=4)
    fig_name = 'test_batch_example'
    dataloader_example(test_loader, path_fig_dir, fig_name, batch_size=batch_size, plot_to=4)

    log_dir = os.path.join(model_dir, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path_settings = os.path.join(log_dir, 'settings.txt')
    with open(path_settings, 'w') as f:
        f.write(f'Pretrained: {pretrained}\n')
        f.write(f'Model Version: {model_version}\n')
        f.write(f'Batch Size: {batch_size}\n')
        f.write(f'Image Size: {h}x{w}\n')
        f.write(f'Margin: {margin}\n')
        f.write(f'Learning Rate: {lr}\n')
        f.write(f'Step Size: {step_size}\n')
        f.write(f'Gamma: {gamma}\n')
        f.write(f'Number of Epochs: {num_epochs}\n')
        f.write(f'Unfreeze Epoch: {unfreeze_n_epoch}\n')
        f.write(f'Unfreeze All Epoch: {unfreeze_all_epoch}\n')

    path_train_learning_logs = os.path.join(log_dir, 'train_learning_logs.csv')
    if not os.path.exists(path_train_learning_logs):
        print(f'Creating file: {path_train_learning_logs}')
        df_train_learning_logs = pd.DataFrame(columns=['epoch', 'iteration', 'loss'])
        df_train_learning_logs.to_csv(path_train_learning_logs, index=False)

    path_test_learning_logs = os.path.join(log_dir, 'test_learning_logs.csv')
    if not os.path.exists(path_test_learning_logs):
        print(f'Creating file: {path_test_learning_logs}')
        df_test_learning_logs = pd.DataFrame(columns=['epoch', 'iteration', 'loss'])
        df_test_learning_logs.to_csv(path_test_learning_logs, index=False)

    best_test_loss = float('inf')

    print("Starting the training loop...")
    for epoch in range(num_epochs):

        # unfreeze layers
        if epoch == unfreeze_n_epoch:
            print(f"Unfreezing all layers at epoch {epoch}")
            model.unfreeze_last_n_layers(n_unfreeze)
        if epoch == unfreeze_all_epoch:
            print(f"Unfreezing all layers at epoch {epoch}")
            model.unfreeze_all()

        model.train()
        train_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
        for batch_idx, batch in enumerate(train_loader):
            (anchor_img, positive_img, negative_img), _ = batch
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            optimizer.zero_grad()
            anchor_emb = model(anchor_img)
            positive_emb = model(positive_img)
            negative_emb = model(negative_img)

            loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            ## lazily log learning
            df_new_row = pd.DataFrame({'epoch': [epoch], 'iteration': [batch_idx], 'loss': [loss.item()]})
            df_train_learning_logs = pd.concat([df_train_learning_logs, df_new_row], ignore_index=True)

            pbar.set_description(f'Epoch ({epoch + 1}/{num_epochs}) Train Loss: {loss.item()}')
            pbar.update(1)
        pbar.close()

        train_loss /= len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}")

        df_train_learning_logs.to_csv(path_train_learning_logs, index=False)

        # Evaluation
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader), position=0, leave=True)
            for batch in test_loader:
                (anchor_img, positive_img, negative_img), _ = batch
                anchor_img = anchor_img.to(device)
                positive_img = positive_img.to(device)
                negative_img = negative_img.to(device)

                anchor_emb = model(anchor_img)
                positive_emb = model(positive_img)
                negative_emb = model(negative_img)

                loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
                test_loss += loss.item()

                ## lazily log learning
                df_new_row = pd.DataFrame({'epoch': [epoch], 'iteration': [batch_idx], 'loss': [loss.item()]})
                df_test_learning_logs = pd.concat([df_test_learning_logs, df_new_row], ignore_index=True)

                pbar.set_description(f'Epoch ({epoch + 1}/{num_epochs}) Test Loss: {loss.item()}')
                pbar.update(1)
            pbar.close()

        test_loss /= len(test_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}")

        if test_loss < best_test_loss:
            print(f"Saving the best model with test loss: {test_loss:.4f}")
            best_test_loss = test_loss
            torch.save(model.state_dict(), path_best_model)

        df_test_learning_logs.to_csv(path_test_learning_logs, index=False)

        ## visualize learning logs
        visualize_learning_logs(path_train_learning_logs, path_test_learning_logs, log_dir)

        # Save the model
        torch.save(model.state_dict(), path_model)