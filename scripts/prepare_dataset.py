import os
from dotenv import load_dotenv
import boto3
load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME = os.getenv('BUCKET_NAME')
BUCKET_PREFIX = os.getenv('BUCKET_PREFIX')
LOCATION_DIR = os.getenv('LOCATION_DIR')
OUT_DATA_FOLDER = os.getenv('OUT_DATA_FOLDER')

def download_data_set(AWS_ACCESS_KEY_ID:str, AWS_SECRET_ACCESS_KEY:str, BUCKET_NAME:str, BUCKET_PREFIX:str, LOCATION_DIR:str) -> None:
    """
    Download and save datasets from AWS s3 bucket. 

    First, before executing, create a folder named 'data' in the root/project directory.

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
    LOCATION_DIR : str
        Directory.
    Returns
    -------
        None. It downloads and saves the files into 'data' folder (previously created)
    """

    s3_resource = boto3.resource('s3',aws_access_key_id=AWS_ACCESS_KEY_ID,aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    bucket = s3_resource.Bucket(BUCKET_NAME)
    
    for object_summary in bucket.objects.filter(Prefix=BUCKET_PREFIX):
        try:
            with open(os.path.join(
                    '..',LOCATION_DIR, os.path.split(object_summary.key)[-1]
                    ), 'wb') as data:
                bucket.download_fileobj(object_summary.key, data)
        except IsADirectoryError:
            continue

def walkdir(folder):
    """
    Walk through all the files in a directory and its subfolders.

    Parameters
    ----------
    folder : str
        Path to the folder you want to walk.

    Returns
    -------
        For each file found, yields a tuple having the path to the file
        and the file name.
    """
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            yield (dirpath, filename)



def split_data_set(data_folder, output_data_folder):
    
    data_folder=os.getenv('LOCATION_DIR')
    output_data_folder=os.getenv('OUT_DATA_FOLDER')
    list=walkdir(data_folder)

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
                # Crear la carpeta data_v1
            if not os.path.exists(output_data_folder):
                os.makedirs(output_data_folder)
                print(output_data_folder)
                
                # Crear la carpeta test/train/val
                
            subset_folder_path=os.path.join(output_data_folder,subset)
            
            
            #Link the image to dst
            if not os.path.exists(subset_folder_path):
                # if the folder directory is not present 
                # then create it.
                os.makedirs(subset_folder_path)
                print(subset_folder_path)
            
            src= os.path.join( data_folder,filename) # data/train_02.jpg
            dst=os.path.join(subset_folder_path , filename) # data/data_v1/train/train_02.jpg
                
            if not os.path.exists(dst):
                # Move file
                os.link(src, dst)
                print(dst)

