import multiprocessing
import pandas as pd
import os
from functools import partial
from packages.prepare_dataset import get_list_of_filenames


AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv('BUCKET_NAME')
BUCKET_PREFIX = os.getenv('BUCKET_PREFIX')

DATA_FOLDER = os.getenv('LOCATION_DIR') # data_downloaded
OUTPUT_DATA_FOLDER = os.getenv('OUTPUT_DATA_FOLDER')
OUTPUT_DATA_BB_FOLDER = os.getenv('OUTPUT_DATA_BB_FOLDER')

LABELS_FOLDER = os.getenv('LABELS_FOLDER')
IMAGES_FOLDER = os.getenv('IMAGES_FOLDER')

def create_labels_dir(OUTPUT_DATA_FOLDER:str , LABELS_FOLDER:str, subset: str):

    # create path: data/labels
    labels_folder_path = os.path.join(OUTPUT_DATA_FOLDER, LABELS_FOLDER)

    if not os.path.exists(labels_folder_path):
        os.makedirs(labels_folder_path)

    # create path: data/labels/train
    subset_labels_folder_path = os.path.join(labels_folder_path, subset)

    if not os.path.exists(subset_labels_folder_path):
        os.makedirs(subset_labels_folder_path)

def from_csv_to_list(filename: str, DATA_FOLDER:str, subset: str):

    df_annotations = pd.read_csv(f'{DATA_FOLDER}/annotations_{subset}.csv', names=["image_name", "x1", "y1", "x2", "y2","class", "image_width", "image_height"])

    list_normalize_coord = []
    starter = 0

    for i in df_annotations.loc[df_annotations['image_name'] == filename].values:
        
        b_center_x = (i[1] + i[3]) / 2 
        b_center_y = (i[2] + i[4]) / 2
        b_width    = (i[3] - i[1])
        b_height   = (i[4] - i[2])

        # Normalise the co-ordinates by the dimensions of the image
        image_w = i[6]
        image_h= i[7]
        image_c = i[5]
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h
        
        starter += 1
    
        list_normalize_coord.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(starter, b_center_x, b_center_y, b_width, b_height))
    
    return list_normalize_coord

def get_txt_normalized_coords(subset:str, subset_labels_folder_path:str, filename:str):

    list_normalize_coord = from_csv_to_list(filename= filename, DATA_FOLDER= DATA_FOLDER, subset= subset)

    path_to_file = os.path.join(subset_labels_folder_path, filename).replace("jpg", "txt")

    with open(path_to_file, 'w') as f:
        f.write("\n".join(list_normalize_coord))

def run():

    # Generate TXT with normalized bounding box coordinates & create each label directory (data/labels/train-test-val)

    labels_folder_path = os.path.join(OUTPUT_DATA_FOLDER, LABELS_FOLDER)

    subset_train = 'train'
    create_labels_dir(OUTPUT_DATA_FOLDER = OUTPUT_DATA_FOLDER, LABELS_FOLDER = LABELS_FOLDER, subset= subset_train)
    list_of_filenames_train = get_list_of_filenames(DATA_FOLDER= DATA_FOLDER, subset= subset_train)
    pool = multiprocessing.Pool()
    subset_labels_folder_path = os.path.join(labels_folder_path, subset_train)
    func = partial(get_txt_normalized_coords, subset_train, subset_labels_folder_path)
    pool.map(func, list_of_filenames_train)
    pool.close()
    pool.join()

    subset_val = 'val'
    create_labels_dir(OUTPUT_DATA_FOLDER = OUTPUT_DATA_FOLDER, LABELS_FOLDER = LABELS_FOLDER, subset= subset_val)
    list_of_filenames_val = get_list_of_filenames(DATA_FOLDER= DATA_FOLDER, subset= subset_val)
    pool = multiprocessing.Pool()
    subset_labels_folder_path = os.path.join(labels_folder_path, subset_val)
    func = partial(get_txt_normalized_coords, subset_val, subset_labels_folder_path)
    pool.map(func, list_of_filenames_val)
    pool.close()
    pool.join()


    subset_test = 'test'
    create_labels_dir(OUTPUT_DATA_FOLDER = OUTPUT_DATA_FOLDER, LABELS_FOLDER = LABELS_FOLDER, subset= subset_test)
    list_of_filenames_test = get_list_of_filenames(DATA_FOLDER= DATA_FOLDER, subset= subset_test)
    pool = multiprocessing.Pool()
    subset_labels_folder_path = os.path.join(labels_folder_path, subset_test)
    func = partial(get_txt_normalized_coords, subset_test, subset_labels_folder_path)
    pool.map(func, list_of_filenames_test)
    pool.close()
    pool.join()

def main_prepare_labels():
    run()

if __name__ == '__main__':
    main_prepare_labels()  