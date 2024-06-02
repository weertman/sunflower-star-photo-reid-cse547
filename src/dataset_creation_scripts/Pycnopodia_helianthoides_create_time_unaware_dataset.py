import os
import torch
import datetime
import ultralytics
from ultralytics import YOLO
import glob
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import umap
from sklearn.decomposition import PCA

def process_image_file (image_file, target_class, model, yolo_classes, conf=0.25):
    image = cv2.imread(image_file)
    results = model(image, conf=conf, verbose=False)
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(e, image_file)
        return None

    mask_areas = []
    object_ids = []
    masks = []
    for result in results:
        if result.masks is None:
            continue

        for mask, box in zip(result.masks.xy, result.boxes):
            points = np.int32([mask])
            mask_area = cv2.contourArea(points)
            mask_areas.append(mask_area)
            object_id = yolo_classes[int(box.cls[0].item())]
            object_ids.append(object_id)

            masks.append(mask)

    ## get the index of largest mask area where the object_id is 'Pycnopodia_helianthoides'
    indexes = [i for i, x in enumerate(object_ids) if x == target_class]
    if len(indexes) > 0:
        index = indexes[np.argmax([mask_areas[i] for i in indexes])]
        mask = masks[index]
        ## mask and crop the raw image using the mask so that we can only see the object of interest
        mask = np.int32([mask])
        mask_image = np.zeros_like(image)
        cv2.fillPoly(mask_image, mask, (255, 255, 255))
        ## apply the mask to the raw image
        mask_image = cv2.bitwise_and(image, mask_image)
        ## crop the raw image using the mask
        x, y, w, h = cv2.boundingRect(mask)
        mask_image = mask_image[y:y+h, x:x+w]
        ## center masked image on square canvas
        max_dim = max(mask_image.shape)
        mask_image = cv2.copyMakeBorder(mask_image, (max_dim - mask_image.shape[0]) // 2, (max_dim - mask_image.shape[0]) // 2,
                                        (max_dim - mask_image.shape[1]) // 2, (max_dim - mask_image.shape[1]) // 2,
                                        cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return mask_image
    else:
        return None


def process_df (df, df_dir, target_class, model, yolo_classes, conf=0.25,
                target_size=(512, 512)):
    print(f'Processing {len(df)} images')
    if not os.path.exists(df_dir):
        print(f'Creating directory: {df_dir}')
        os.makedirs(df_dir)

    n_images_per_id = {}
    path_processed_images = []
    masked_images = []

    pbar = tqdm(total=len(df), desc='Processing images', position=0, leave=True)
    for _, row in df.iterrows():
        id_code = row['id_code']
        id_dir = os.path.join(df_dir, id_code)
        if not os.path.exists(id_dir):
            os.makedirs(id_dir)

        image_file = row['image_file']
        mask_image = process_image_file(image_file, target_class, model, yolo_classes, conf)
        if mask_image is not None:
            mask_image_file = os.path.join(id_dir, os.path.basename(image_file))
            mask = cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR)
            if target_size is not None:
                mask = cv2.resize(mask, target_size)
            cv2.imwrite(mask_image_file, mask)
            path_processed_images.append(mask_image_file)
            masked_images.append(mask)
            if id_code in n_images_per_id:
                n_images_per_id[id_code] += 1
            else:
                n_images_per_id[id_code] = 1
        else:
            masked_images.append(None)
        pbar.update(1)
    pbar.close()
    return n_images_per_id, path_processed_images, masked_images

def pycnopodia_helianthoides():
    create_n_datasets = 3

    for n in range(create_n_datasets):
        print('PyTorch version: ', torch.__version__)
        # check if CUDA is available
        print(f'CUDA version: {torch.version.cuda}')
        # check device
        print(torch.cuda.current_device())

        print(f'Ultralytics version: {ultralytics.__version__}')
        print(ultralytics.checks())

        target_class = 'Pycnopodia_helianthoides'

        dst_data_dir_head = f'{target_class}_full_'

        src_data_root = r'C:\Users\wlwee\OneDrive\Desktop\sorted_stars'
        src_image_files = glob.glob(os.path.join(src_data_root, '*', '*', '*', '*.*'))

        date_txt = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        dst_data_dir = os.path.join('../..', 'data', dst_data_dir_head + 'event-unaware_training-split_' + date_txt)
        if not os.path.exists(dst_data_dir):
            os.makedirs(dst_data_dir)
            print('Directory created: ', dst_data_dir)

        path_model = os.path.join('../..', 'models', 'yolov8', 'best.pt')
        if not os.path.exists(path_model):
            print('Model not found: ', path_model)
            raise FileNotFoundError
        model = YOLO(path_model)
        model.to('cuda:1')

        yolo_classes = list(model.names.values())
        classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

        print(f'Classes: {yolo_classes}')
        print(f'Classes IDs: {classes_ids}')

        ## magenta
        color_other_species = (255, 0, 255)
        ## blue
        color_pycnopodia = (255, 0, 0)
        colors = [color_other_species, color_pycnopodia]

        for image_file in src_image_files[:5]:
            print(image_file)
            group_code = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_file))))
            id_code = os.path.basename(os.path.dirname(os.path.dirname(image_file))) + '__' + group_code
            event_code = os.path.basename(os.path.dirname(image_file))
            print(f'\tgroup_code: {group_code}')
            print(f'\tid_code: {id_code}')
            print(f'\tevent_code: {event_code}')

        print()
        print(f'Total number of images: {len(src_image_files)}')

        print(f'Creating dataframe...')
        id_codes = []
        image_events = []
        group_codes = []
        for image_file in src_image_files:
            group_code = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_file))))
            id_code = os.path.basename(os.path.dirname(os.path.dirname(image_file))) + '__' + group_code
            event_code = os.path.basename(os.path.dirname(image_file))

            id_codes.append(id_code)
            image_events.append(event_code)
            group_codes.append(group_code)

        df = pd.DataFrame(
            {'id_code': id_codes, 'group_code': group_codes, 'event_code': image_events, 'image_file': src_image_files})
        print(df.head())
        print(df.shape)

        ## randomly sample 5 rows from df
        df_random = df.copy().sample(5)

        conf = 0.25
        print(f'Creating example mask images...')
        for i, row in df_random.iterrows():
            path_image = row['image_file']
            print(f'Processing image: {path_image}')
            name_image = os.path.basename(path_image).split('.')[0]
            mask_image = process_image_file(path_image, target_class, model, yolo_classes, conf=conf)
            if mask_image is None:
                print(f'No mask found for {path_image}')
                continue
            path_image = os.path.join(dst_data_dir, f'{i}_{name_image}_mask_example.png')
            cv2.imwrite(path_image, cv2.cvtColor(mask_image, cv2.COLOR_RGB2BGR))
        del df_random

        unique_id_codes = df['id_code'].unique()

        train_dfs = []
        test_dfs = []

        print(f'Creating training and test datasets...')

        for id_code in unique_id_codes[:]:
            df_id_code = df[df['id_code'] == id_code]
            print(f'id_code: {id_code}', f'number of images: {len(df_id_code)}')

            if len(df_id_code) < 2:
                print(f'\t WARNING id_code: {id_code}', f'number of images: {len(df_id_code)}')
                continue

            df_train, df_test = train_test_split(df_id_code, test_size=0.2)

            if len(df_train) > 1:
                train_dfs.append(df_train)
                test_dfs.append(df_test)
                print(f'\ttrain: {len(df_train)}', f'test: {len(df_test)}')
            else:
                df_train, df_test = train_test_split(df_id_code, test_size=0.2)
                if len(df_train) > 1:
                    train_dfs.append(df_train)
                    test_dfs.append(df_test)
                    print(f'\tREROLL train: {len(df_train)}', f'test: {len(df_test)}')
                else:
                    print(f'\tWARNING id_code: {id_code}', f'number of images: {len(df_id_code)}')

        ## concatenate the dataframes
        df_train = pd.concat(train_dfs)
        df_test = pd.concat(test_dfs)

        print(f'Training dataset: {df_train.shape}')
        print(f'df_train.head() = {df_train.head()}')
        print(f'Test dataset: {df_test.shape}')
        print(f'df_test.head() = {df_test.head()}')

        train_dir = os.path.join(dst_data_dir, 'train')
        n_images_per_id_train, path_train_images, masked_train_images = process_df(df_train, train_dir, target_class, model, yolo_classes)

        test_dir = os.path.join(dst_data_dir, 'test')
        n_images_per_id_test, path_test_images, masked_test_images = process_df(df_test, test_dir, target_class, model, yolo_classes)

        print(f'Number of images per id in training dataset: {n_images_per_id_train}')
        print(f'Number of images per id in test dataset: {n_images_per_id_test}')

        print(f'Number of training images: {len(path_train_images)}')
        print(f'Number of test images: {len(path_test_images)}')

        df_train['masked_image'] = masked_train_images
        df_test['masked_image'] = masked_test_images

        ## drop rows with None values
        df_train = df_train.dropna()
        df_test = df_test.dropna()

        path_test_df = os.path.join(dst_data_dir, 'df_test.pickle')
        df_test.to_pickle(path_test_df)
        print(f'Saved test dataframe: {path_test_df}')

        path_train_df = os.path.join(dst_data_dir, 'df_train.pickle')
        df_train.to_pickle(path_train_df)
        print(f'Saved train dataframe: {path_train_df}')


if __name__ == '__main__':
    pycnopodia_helianthoides()

