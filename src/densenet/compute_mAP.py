import torch
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from src.densenet.densenet import CustomDenseNet
from src.densenet.runtime_transforms import RuntimeTransforms
import os
import glob
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def compute_mAP_for_ks_per_id(query_features, gallery_features, query_ids, gallery_ids, k_values):
    """
    Compute the mean Average Precision (mAP) for multiple specified k values for a given set of query and gallery features and their IDs.
    The mAP is first calculated per ID and then averaged across IDs.

    Parameters:
    - query_features (np.ndarray): Feature vectors of the query set (shape: [num_queries, feature_dim])
    - gallery_features (np.ndarray): Feature vectors of the gallery set (shape: [num_gallery, feature_dim])
    - query_ids (np.ndarray): IDs corresponding to the query set (shape: [num_queries])
    - gallery_ids (np.ndarray): IDs corresponding to the gallery set (shape: [num_gallery])
    - k_values (list of int): List of k values at which to calculate mAP

    Returns:
    - dict: A dictionary where the keys are k values and the values are dictionaries with mAP per ID and overall mAP.
            Example: {k: {'per_id': {id1: mAP1, id2: mAP2, ...}, 'overall': overall_mAP}}
    """

    unique_ids = np.unique(query_ids)
    distances = pairwise_distances(query_features, gallery_features, metric='euclidean')

    mAP_results = {k: {'per_id': {}, 'overall': 0} for k in k_values}

    for id_ in unique_ids:
        query_indices = np.where(query_ids == id_)[0]
        if len(query_indices) == 0:
            continue

        precisions_at_k = {k: [] for k in k_values}

        for i in query_indices:
            query_id = query_ids[i]
            dists = distances[i]
            sorted_indices = np.argsort(dists)
            sorted_gallery_ids = gallery_ids[sorted_indices]

            for k in k_values:
                if query_id in sorted_gallery_ids[:k]:
                    precisions_at_k[k].append(1)
                else:
                    precisions_at_k[k].append(0)

        for k in k_values:
            mAP_results[k]['per_id'][id_] = np.mean(precisions_at_k[k])

    for k in k_values:
        mAP_results[k]['overall'] = np.mean(list(mAP_results[k]['per_id'].values()))

    return mAP_results


def print_mAP_by_id_results(mAP_results, num_dict):
    """
    Print the mAP results for different k values.

    Parameters:
    - mAP_results (dict): A dictionary where the keys are k values and the values are dictionaries with mAP per ID and overall mAP.
                          Example: {k: {'per_id': {id1: mAP1, id2: mAP2, ...}, 'overall': overall_mAP}}
    - num_dict (dict): id num to id string dictionary
    """

    for k, results in mAP_results.items():
        print(f"Results for k = {k}:")
        print("Per ID mAP:")
        for id_, mAP in results['per_id'].items():
            id_str = num_dict[id_]
            print(f"k={k}  ID {id_str}: mAP = {mAP:.4f}")
        print(f"Overall mAP when running by ID is: {results['overall']:.4f}")
        print("-" * 40)
        break


def extract_location_code(id_string):
    """
    Extract the location code from the ID string.

    Parameters:
    - id_string (str): The ID string in the format 'name__location_code'

    Returns:
    - str: The location code
    """
    return id_string.split('__')[1]

def visualize_and_save_mAP_at_1_results(mAP_results, target_directory, num_dict):
    """
    Visualize and save the mAP@1 results.

    Parameters:
    - mAP_results (dict): A dictionary where the keys are k values and the values are dictionaries with mAP per ID and overall mAP.
                          Example: {k: {'per_id': {id1: mAP1, id2: mAP2, ...}, 'overall': overall_mAP}}
    - target_directory (str): The directory where the plots will be saved.
    - num_dict (dict): Dictionary mapping ID numbers to ID strings for plotting.
    """

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    k = 1  # We are focusing on mAP@1
    if k not in mAP_results:
        print(f"No results for k={k}")
        return

    # Extract mAP per ID and sort by mAP values (smallest to largest)
    ids = list(mAP_results[k]['per_id'].keys())
    mAP_per_id = mAP_results[k]['per_id']
    sorted_ids = sorted(ids, key=lambda id_: mAP_per_id[id_])

    # Overall mAP
    overall = mAP_results[k]['overall']

    # Extract location codes and assign colors
    location_codes = [extract_location_code(num_dict[id_]) for id_ in sorted_ids]
    unique_locations = sorted(set(location_codes))
    cmap = cm.get_cmap('tab20', len(unique_locations))
    color_map = {location: cmap(i) for i, location in enumerate(unique_locations)}
    colors = [color_map[location] for location in location_codes]

    # Determine the figure size based on the number of IDs
    plt.figure(figsize=(max(14, len(sorted_ids) * 0.3), 7))

    id_strings = [num_dict[id_] for id_ in sorted_ids]
    mAP_values = [mAP_per_id[id_] for id_ in sorted_ids]

    plt.scatter(id_strings, mAP_values, color=colors)

    plt.xlabel('ID')
    plt.ylabel('mAP@1')
    plt.title(f'mAP@1 per ID - Overall by ID mAP@1: {overall:.4f}')
    plt.xticks(rotation=45, ha='right')

    ax = plt.gca()
    for ticklabel, tickcolor in zip(ax.get_xticklabels(), colors):
        ticklabel.set_color(tickcolor)

    # Turn off the background grid
    ax.grid(False)

    # Turn off the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(target_directory, 'mAP_at_1_per_id.png'), dpi=300, bbox_inches='tight')
    plt.close()

def compute_mAP_for_ks_numpy(query_features, gallery_features, query_ids, gallery_ids, k_values):
    """
    Compute the mean Average Precision (mAP) for multiple specified k values for a given set of query and gallery features and their IDs.

    Parameters:
    - query_features (np.ndarray): Feature vectors of the query set (shape: [num_queries, feature_dim])
    - gallery_features (np.ndarray): Feature vectors of the gallery set (shape: [num_gallery, feature_dim])
    - query_ids (np.ndarray): IDs corresponding to the query set (shape: [num_queries])
    - gallery_ids (np.ndarray): IDs corresponding to the gallery set (shape: [num_gallery])
    - k_values (list of int): List of k values at which to calculate mAP

    Returns:
    - dict: A dictionary where the keys are k values and the values are the corresponding mAP@k
    """

    # Calculate pairwise distances between query and gallery features
    distances = pairwise_distances(query_features, gallery_features, metric='euclidean')

    # Initialize dictionary to store precision at each k
    precisions_at_k = {k: [] for k in k_values}

    # Loop over each query
    for i in range(len(query_features)):
        query_id = query_ids[i]
        dists = distances[i]

        # Sort gallery images by distance
        sorted_indices = np.argsort(dists)
        sorted_gallery_ids = gallery_ids[sorted_indices]

        # Calculate if the correct ID is within the top k results
        for k in k_values:
            if query_id in sorted_gallery_ids[:k]:
                precisions_at_k[k].append(1)
            else:
                precisions_at_k[k].append(0)

    # Calculate mean Average Precision for each k
    mAPs = {k: np.mean(precisions_at_k[k]) for k in k_values}

    return mAPs

def compute_mAP_for_ks_pytorch(query_features, gallery_features, query_ids, gallery_ids, k_values, device='cpu'):
    """
    Compute the mean Average Precision (mAP) for multiple specified k values for a given set of query and gallery features and their IDs.

    Parameters:
    - query_features (torch.Tensor): Feature vectors of the query set (shape: [num_queries, feature_dim])
    - gallery_features (torch.Tensor): Feature vectors of the gallery set (shape: [num_gallery, feature_dim])
    - query_ids (torch.Tensor): IDs corresponding to the query set (shape: [num_queries])
    - gallery_ids (torch.Tensor): IDs corresponding to the gallery set (shape: [num_gallery])
    - k_values (list of int): List of k values at which to calculate mAP
    - device (str): Device to use for computation ('cpu' or 'cuda')

    Returns:
    - dict: A dictionary where the keys are k values and the values are the corresponding mAP@k
    """

    # Move data to the specified device
    query_features = query_features.to(device)
    gallery_features = gallery_features.to(device)
    query_ids = query_ids.to(device)
    gallery_ids = gallery_ids.to(device)

    # Calculate pairwise distances between query and gallery features
    distances = torch.cdist(query_features, gallery_features)

    # Initialize dictionary to store precision at each k
    precisions_at_k = {k: [] for k in k_values}

    # Loop over each query
    for i in range(len(query_features)):
        query_id = query_ids[i].item()
        dists = distances[i]

        # Sort gallery images by distance
        sorted_indices = torch.argsort(dists)
        sorted_gallery_ids = gallery_ids[sorted_indices]

        # Calculate if the correct ID is within the top k results
        for k in k_values:
            if query_id in sorted_gallery_ids[:k]:
                precisions_at_k[k].append(1)
            else:
                precisions_at_k[k].append(0)

    # Calculate mean Average Precision for each k
    mAPs = {k: np.mean(precisions_at_k[k]) for k in k_values}

    return mAPs

def generate_test_data(num_queries=100, num_gallery=1000, feature_dim=512, num_identities=50, noise_level=10.0, seed=42):
    """
    Generate a test dataset with feature vectors and IDs using Gaussian blobs.

    Parameters:
    - num_queries (int): Number of query images
    - num_gallery (int): Number of gallery images
    - feature_dim (int): Dimension of the feature vectors
    - num_identities (int): Number of distinct identities
    - noise_level (float): Noise level for feature vectors
    - seed (int): Random seed for reproducibility

    Returns:
    - query_features (np.ndarray): Generated feature vectors for the query set
    - gallery_features (np.ndarray): Generated feature vectors for the gallery set
    - query_ids (np.ndarray): Generated IDs for the query set
    - gallery_ids (np.ndarray): Generated IDs for the gallery set
    """
    np.random.seed(seed)
    # Initialize base features for each identity using Gaussian blobs
    centers = np.random.rand(num_identities, feature_dim) * 10  # spread out the centers
    query_features = []
    query_ids = []
    gallery_features = []
    gallery_ids = []

    for i in range(num_identities):
        query_features.append(np.random.randn(num_queries // num_identities, feature_dim) * noise_level + centers[i])
        query_ids.extend([i] * (num_queries // num_identities))
        gallery_features.append(np.random.randn(num_gallery // num_identities, feature_dim) * noise_level + centers[i])
        gallery_ids.extend([i] * (num_gallery // num_identities))

    query_features = np.vstack(query_features)
    gallery_features = np.vstack(gallery_features)
    query_ids = np.array(query_ids)
    gallery_ids = np.array(gallery_ids)

    return query_features, gallery_features, query_ids, gallery_ids

def generate_test_data_torch(num_queries=100, num_gallery=1000, feature_dim=512, num_identities=50, noise_level=10.0, device='cpu', seed=42):
    """
    Generate a test dataset with feature vectors and IDs using Gaussian blobs.

    Parameters:
    - num_queries (int): Number of query images
    - num_gallery (int): Number of gallery images
    - feature_dim (int): Dimension of the feature vectors
    - num_identities (int): Number of distinct identities
    - noise_level (float): Noise level for feature vectors
    - device (str): Device to use for computation ('cpu' or 'cuda')
    - seed (int): Random seed for reproducibility

    Returns:
    - query_features (torch.Tensor): Generated feature vectors for the query set
    - gallery_features (torch.Tensor): Generated feature vectors for the gallery set
    - query_ids (torch.Tensor): Generated IDs for the query set
    - gallery_ids (torch.Tensor): Generated IDs for the gallery set
    """
    # Use numpy to generate data for consistency
    query_features_np, gallery_features_np, query_ids_np, gallery_ids_np = generate_test_data(num_queries, num_gallery, feature_dim, num_identities, noise_level, seed)

    # Convert numpy arrays to PyTorch tensors
    query_features = torch.tensor(query_features_np, dtype=torch.float).to(device)
    gallery_features = torch.tensor(gallery_features_np, dtype=torch.float).to(device)
    query_ids = torch.tensor(query_ids_np, dtype=torch.long).to(device)
    gallery_ids = torch.tensor(gallery_ids_np, dtype=torch.long).to(device)

    return query_features, gallery_features, query_ids, gallery_ids

if __name__ == '__main__':
    # Set random seeds for reproducibility
    set_random_seeds()

    # Generate data using numpy
    query_features_np, gallery_features_np, query_ids_np, gallery_ids_np = generate_test_data()

    # Generate data using PyTorch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    query_features_torch, gallery_features_torch, query_ids_torch, gallery_ids_torch = generate_test_data_torch(device=device)

    # Check consistency between numpy and PyTorch generated data
    print('Verifying that mAP works on synthetic data...')
    print("Checking consistency between numpy and PyTorch generated data...")
    print("Query features close:", np.allclose(query_features_np, query_features_torch.cpu().numpy(), atol=1e-6))
    print("Gallery features close:", np.allclose(gallery_features_np, gallery_features_torch.cpu().numpy(), atol=1e-6))
    print("Query IDs close:", np.array_equal(query_ids_np, query_ids_torch.cpu().numpy()))
    print("Gallery IDs close:", np.array_equal(gallery_ids_np, gallery_ids_torch.cpu().numpy()))

    # Compute mAP using numpy implementation
    k_values = [1, 5, 10]
    mAPs_np = compute_mAP_for_ks_numpy(query_features_np, gallery_features_np, query_ids_np, gallery_ids_np, k_values)

    # Compute mAP using PyTorch implementation
    mAPs_torch = compute_mAP_for_ks_pytorch(query_features_torch, gallery_features_torch, query_ids_torch, gallery_ids_torch, k_values, device=device)

    # Print mAP for specified k values
    print("\nNumpy mAP:")
    for k in k_values:
        print(f"mAP@{k}: {mAPs_np[k]:.4f}")

    print("\nPyTorch mAP:")
    for k in k_values:
        print(f"mAP@{k}: {mAPs_torch[k]:.4f}")

    print('Ok, it works on the synthetic data!')

    print()
    print('Now testing mAP on real data using one of our datasets...')
    print()

    path_model = 'best_densenet121__PT-True.pth'

    # Instantiate the model
    model = CustomDenseNet(version='densenet121', pretrained=False, embedding_dim=256, num_embed_layers=3)  # Set pretrained=False as we'll load our weights

    # Load the trained weights
    model.load_state_dict(torch.load(path_model))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Check if the model has been loaded correctly
    model.eval()

    # Gallery dataset path
    path_gallery = r'..\..\data\Pycnopodia_helianthoides_full_event-unaware_training-split_2024-05-03-18-41\train'
    print(f'Gallery dataset path: {path_gallery} {os.path.exists(path_gallery)}')
    ## each subdirectory in the gallery is a identity class
    ## each subdirectory contains images of that identity
    ## the subdirectory name is the identity
    ## the images are in png format
    ## we need to make a code to num dictionary
    ## we need to load each image and pass it through the model to get the embeddings
    ## we need to store the embeddings in a list of length equal to the number of images
    ## we need to store the identity in a list of length equal to the number of images

    # Query dataset path
    path_query = r'..\..\data\Pycnopodia_helianthoides_full_event-unaware_training-split_2024-05-03-18-41\test'
    print(f'Query dataset path: {path_query} {os.path.exists(path_query)}')
    ## each subdirectory in the gallery is a identity class
    ## each subdirectory contains images of that identity
    ## the subdirectory name is the identity
    ## the images are in png format
    ## we need to make a code to num dictionary
    ## these need to be the same as the gallery dataset
    ## we need to load each image and pass it through the model to get the embeddings
    ## we need to store the embeddings in a list of length equal to the number of images
    ## we need to store the identity in a list of length equal to the number of images

    def get_id (path):
        return os.path.basename(os.path.dirname(path))

    path_gallery_images = glob.glob(os.path.join(path_gallery, '*/*.png'))
    gallery_ids = [get_id(path) for path in path_gallery_images]

    path_query_images = glob.glob(os.path.join(path_query, '*/*.png'))
    query_ids = [get_id(path) for path in path_query_images]

    ids_dict = {id: i for i, id in enumerate(sorted(list(set(gallery_ids + query_ids))))}
    num_dict = {i: id for i, id in enumerate(sorted(list(set(gallery_ids + query_ids))))}

    gallery_ids = [ids_dict[id] for id in gallery_ids]
    query_ids = [ids_dict[id] for id in query_ids]

    # Initialize the runtime transforms
    runtime_transforms = RuntimeTransforms(h=256, w=256, resize_ratio=0.5)

    gallery_embeddings = []
    pbar = tqdm(total=len(path_gallery_images), position=0, leave=True)
    for path in path_gallery_images:
        img = Image.open(path).convert("RGB")
        transformed_images = runtime_transforms.apply_transforms(img) ## returns a list of transformed images

        # Embed all the transformed images at once as a batch
        batch = torch.stack([transformed_image for transformed_image in transformed_images]).to(device)
        with torch.no_grad():
            embeddings = model(batch).cpu().numpy()
        # Average the embeddings
        averaged_embedding = np.mean(embeddings, axis=0)
        gallery_embeddings.append(averaged_embedding)
        pbar.update(1)
    pbar.close()

    query_embeddings = []
    pbar = tqdm(total=len(path_query_images), position=0, leave=True)
    for path in path_query_images:
        img = Image.open(path).convert("RGB")
        transformed_images = runtime_transforms.apply_transforms(img)  # Returns a list of transformed images

        # Embed all the transformed images at once as a batch
        batch = torch.stack([transformed_image for transformed_image in transformed_images]).to(device)
        with torch.no_grad():
            embeddings = model(batch).cpu().numpy()

        # Average the embeddings
        averaged_embedding = np.mean(embeddings, axis=0)
        query_embeddings.append(averaged_embedding)
        pbar.update(1)
    pbar.close()

    # Convert to numpy arrays
    gallery_embeddings = np.array(gallery_embeddings)
    query_embeddings = np.array(query_embeddings)
    gallery_ids = np.array(gallery_ids)
    query_ids = np.array(query_ids)

    # Compute mAP using the computed embeddings
    mAPs_computed = compute_mAP_for_ks_numpy(query_embeddings, gallery_embeddings, query_ids, gallery_ids, k_values)

    print("\nComputed mAP:")
    mAP_dict = {k: mAPs_computed[k] for k in k_values}
    for k in k_values:
        print(f"mAP@{k}: {mAP_dict[k]:.4f}")
    # save mAP dict
    np.save('mAP_dict.npy', mAP_dict)

    ## define a dataframe to store the embeddings
    ## columns will be ['image_path', 'identity', 'gallery/query', 'embedding']
    ## save the dataframe as a pickle file
    dataframe = pd.DataFrame(columns=['image_path', 'identity', 'gallery/query', 'embedding'])
    for i, path in enumerate(path_gallery_images):
        dataframe.loc[i] = [path, gallery_ids[i], 'gallery', gallery_embeddings[i].tolist()]
    for i, path in enumerate(path_query_images):
        dataframe.loc[len(path_gallery_images) + i] = [path, query_ids[i], 'query', query_embeddings[i].tolist()]
    dataframe.to_pickle('embeddings.pkl')

    mAP_results = compute_mAP_for_ks_per_id(query_embeddings, gallery_embeddings, query_ids, gallery_ids, k_values)

    ## print the mAP results
    print_mAP_by_id_results(mAP_results, num_dict)

    # Visualize and save the mAP results
    visualize_and_save_mAP_at_1_results(mAP_results, os.getcwd(), num_dict)

    np.save('mAP_by_id_dict.npy', mAP_dict)




