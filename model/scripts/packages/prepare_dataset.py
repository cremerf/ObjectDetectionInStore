import os
from dotenv import load_dotenv
import boto3
import pandas as pd
import cv2
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv('BUCKET_NAME')
BUCKET_PREFIX = os.getenv('BUCKET_PREFIX')

DATA_FOLDER = os.getenv('LOCATION_DIR') # data_downloaded
OUTPUT_DATA_FOLDER = os.getenv('OUTPUT_DATA_FOLDER')
OUTPUT_DATA_BB_FOLDER = os.getenv('OUTPUT_DATA_BB_FOLDER')

LABELS_FOLDER = os.getenv('LABELS_FOLDER')
IMAGES_FOLDER = os.getenv('IMAGES_FOLDER')


def download_data_set(AWS_ACCESS_KEY_ID:str, AWS_SECRET_ACCESS_KEY:str, BUCKET_NAME:str, BUCKET_PREFIX:str, DATA_FOLDER:str) -> None:
    """
    Download and save datasets from AWS s3 bucket. 

    Creates folder named 'data' if not exists in the root/project directory.

    Parameters
    ----------
    AWS_ACCESS_KEY_ID : str
        ID of your AWS connection.
    AWS_SECRET_ACCESS_KEY : 
        Password provided by AWS
    BUCKET_NAME : str
        Bucket name, used to instantiate s3 class/object.
    BUCKET_PREFIX : str
        Prefix to plug into the desired parent folder from the bucket.
    DATA_FOLDER : str
        Directory.

    Returns
    -------
        None. It downloads and saves the files into 'data' folder (previously created)
    """

    s3_resource = boto3.resource('s3',aws_access_key_id=AWS_ACCESS_KEY_ID,aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    bucket = s3_resource.Bucket(BUCKET_NAME)

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        for object_summary in bucket.objects.filter(Prefix=BUCKET_PREFIX):
            try:

                with open(os.path.join(DATA_FOLDER, os.path.split(object_summary.key)[-1]), 'wb') as data:

                    bucket.download_fileobj(object_summary.key, data)
            except IsADirectoryError:
                continue

def walkdir(DATA_FOLDER: str):
    """
    Walk through all the files in a directory and its subfolders.

    Parameters
    ----------
    DATA_FOLDER : str
        Path to the folder you want to walk.

    Returns
    -------
        For each file found, yields a tuple having the path to the file
        and the file name.
    """
    for dirpath, _, files in os.walk(DATA_FOLDER):
        for filename in files:
            yield (dirpath, filename)



def split_data_set(DATA_FOLDER: str, OUTPUT_DATA_FOLDER: str, IMAGES_FOLDER: str):
    """
    Split train/val/test samples images and save in respective folder. 

    Parameters
    ----------
    DATA_FOLDER : str
        Path to the folder you want to walk.
    OUTPUT_DATA_FOLDER: str
        Folder in which samples will be distributed.

    Returns
    -------
        None.
    """

    list = walkdir(DATA_FOLDER)

    #Create a list of filename folder
    for list_ in list:
        filename=list_[1]
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            if filename.startswith("test"):
                subset='test'
            elif filename.startswith("train"):
                subset='train'
            elif filename.startswith("val"):
                subset='val'

            # Crear la carpeta data
            if not os.path.exists(OUTPUT_DATA_FOLDER):
                os.makedirs(OUTPUT_DATA_FOLDER)
            
            images_folder_path = os.path.join(OUTPUT_DATA_FOLDER, IMAGES_FOLDER)

            # Create data/images if not exists.
            if not os.path.exists(images_folder_path):
                os.makedirs(images_folder_path)

            images_subset_folder_path = os.path.join(images_folder_path, subset)

            # Create data/images/{subset} if not exists.
            if not os.path.exists(images_subset_folder_path):
                os.makedirs(images_subset_folder_path)

            src = os.path.join(DATA_FOLDER, filename) # data/train_02.jpg
            dst = os.path.join(images_subset_folder_path , filename) # data/images/train/train_02.jpg
                
            if not os.path.exists(dst): # y no esta corrupta la imagen -> if not imagen_corrupta:
                # Move file
                os.link(src, dst)


def get_list_of_filenames(DATA_FOLDER: str, subset: str):

    list_of_filenames = []

    list_of_dirs = walkdir(DATA_FOLDER)

    for files in list_of_dirs:
        if files[1].lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            if files[1].startswith(subset):
                print(files[1])
                list_of_filenames.append(files[1])
    
    return list_of_filenames

def plot_bounding_box(DATA_FOLDER: str, OUTPUT_DATA_FOLDER: str, OUTPUT_DATA_BB_FOLDER: str, subset: str):


    data = pd.read_csv(f'{DATA_FOLDER}/annotations_{subset}.csv', names=["image_name", "x1", "y1", "x2", "y2","class", "image_width", "image_height" ])

    # get list of filenames
    filenames = get_list_of_filenames(DATA_FOLDER= DATA_FOLDER, subset= subset)

    # path for data/train
    folder_path = os.path.join(OUTPUT_DATA_FOLDER, subset)
    print(f'Ruta de {OUTPUT_DATA_FOLDER} + {subset} = {folder_path}')

    # path for data_bb
    folder_path_bb = os.path.join(OUTPUT_DATA_BB_FOLDER)
    print(f'Ruta de {OUTPUT_DATA_BB_FOLDER} = {folder_path_bb}')

    # if folder for data_bb is not present, create it.
    if not os.path.exists(folder_path_bb):
        os.mkdir(folder_path_bb)

    # path for data_bb/train
    folder_path_final = os.path.join(folder_path_bb, subset)
    print(f'Ruta de {folder_path_bb} + {subset} = {folder_path_final}')

    # if the folder directory is not present then create it.
    if not os.path.exists(folder_path_final):
        os.mkdir(folder_path_final)

    # loop over filenames to start plotting
    for filename in filenames:

        image_path = os.path.join(folder_path, filename)

        image_path_bb = os.path.join(folder_path_final, filename)

        # if the image path is not present then plot it and save it.
        if os.path.exists(image_path) and not os.path.exists(image_path_bb):

            # get the index for each filenames
            index = data[data['image_name'] == filename].index.tolist()

            # load the image
            image = cv2.imread(image_path)

            # loop over indexes and for each index plot the rectangles
            for idx in index:

                # get the coordinates for each index/rectangle
                x1 = int(data.iloc[idx]['x1'])
                y1 = int(data.iloc[idx]['y1'])
                x2 = int(data.iloc[idx]['x2'])
                y2 = int(data.iloc[idx]['y2'])


                # # Window name in which image is displayed
                window_name = 'Object'
                # represents the top left corner of rectangle
                start_point=(x1, y1)

                # represents the top right corner of rectangle
                end_point=(x2,y2)

                # # Blue color in BGR
                color = (0, 0, 255)

                # # Line thickness of 2 px
                thickness = 3

                # plot the rectangle over the image
                image = cv2.rectangle(image, start_point, end_point, color, thickness)

            # save the img
            cv2.imwrite(image_path_bb, image)

    def resizing_images():


        None

def run():

    download_data_set(
        AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY_ID, 
        AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY, 
        BUCKET_NAME= BUCKET_NAME, 
        BUCKET_PREFIX = BUCKET_PREFIX,
        DATA_FOLDER = DATA_FOLDER)

    # Split dataset into test/train/val
    split_data_set(DATA_FOLDER = DATA_FOLDER, OUTPUT_DATA_FOLDER = OUTPUT_DATA_FOLDER, IMAGES_FOLDER = IMAGES_FOLDER)

    # Plot BB in train samples -> sobre datos limpios
    subset_train = 'train'
    plot_bounding_box(DATA_FOLDER = DATA_FOLDER, OUTPUT_DATA_FOLDER= OUTPUT_DATA_FOLDER, OUTPUT_DATA_BB_FOLDER = OUTPUT_DATA_BB_FOLDER, subset= subset_train)

    # Plot BB in val samples
    subset_val = 'val'
    plot_bounding_box(DATA_FOLDER = DATA_FOLDER, OUTPUT_DATA_FOLDER= OUTPUT_DATA_FOLDER, OUTPUT_DATA_BB_FOLDER = OUTPUT_DATA_BB_FOLDER, subset= subset_val)

    # Plot BB in test samples
    subset_test = 'test'
    plot_bounding_box(DATA_FOLDER = DATA_FOLDER, OUTPUT_DATA_FOLDER= OUTPUT_DATA_FOLDER, OUTPUT_DATA_BB_FOLDER = OUTPUT_DATA_BB_FOLDER, subset= subset_test)


def main_prepare_datasets():
    run()

if __name__ == '__main__':
    main_prepare_datasets()  


