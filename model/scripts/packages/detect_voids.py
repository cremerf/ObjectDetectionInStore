from scripts.packages.yolo_predict import YOLO_Pred
from scripts.packages.paths import ODISPaths 
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import cv2
import multiprocessing
from functools import partial
import itertools
import os
from scripts.packages import settings

PATHS = ODISPaths()

def get_neightbours(df_predictions:pd.DataFrame, neightbour:str, index_it:int) -> dict:
    """
    Identifies the neightbours of each predicted bounding box from a given image.
    
    Takes as parameter a dataframe which stores all the coords of the predicted bounding boxes.
    Defines 4 filters to limit the spectrum to look for the neightbours. 
    
    Each filter is calculated from X_center/Y_center.

    You have to specify which side(right or left) you want to evaluate. 

    Returns a dictionary with key:pair values, where keys are the evaluted bounding box and the pair values are the neightbours of that key value.

    This function is structured in a way to be suited for parallelizing. 
    
    The goal is to optimize times frames and resources to get the left/right neightbours as fast as possible

    Args:
        df_predictions (pd.DataFrame): Dataframe with predicted bb obtained with Yolov5.
        neightbour (str): Right or Left.
        index_it (int): _description_

    Returns:
        dict_of_neightbours: Bounding_box:BB_Neightbours
    """

    dict_of_neightbours = {} 

    X_center_0 = df_predictions.loc[index_it][0]
    Y_center_0 = df_predictions.loc[index_it][1]
    Width_0 = df_predictions.loc[index_it][2]
    height_0 = df_predictions.loc[index_it][3]
    threshold_x_0 = 1 * Width_0
    threshold_y_0 = 1 * height_0

    INPUT_WH_YOLO = 640
    filter_1 = (df_predictions.loc[:, 'X_center'] < (X_center_0 + INPUT_WH_YOLO/2))
    filter_2 = (df_predictions.loc[:, 'X_center'] > (X_center_0 - INPUT_WH_YOLO/2))
    filter_3 = (df_predictions.loc[:, 'Y_center'] > (Y_center_0 - INPUT_WH_YOLO/2))
    filter_4 = (df_predictions.loc[:, 'Y_center'] < (Y_center_0 + INPUT_WH_YOLO/2))
    filter_final = (filter_1 & filter_2) & (filter_3 & filter_4)

    df_predictions_final_2 = df_predictions[filter_final]

    for index_bb in df_predictions_final_2.index:

        list_of_neightbours_l = []
        list_of_neightbours_r = []
        list_of_alones_l = []
        list_of_alones_r = []

        X_center_neightbour = df_predictions.loc[index_bb][0]
        Y_center_neightbour = df_predictions.loc[index_bb][1]
        a = 2
        k = 2

        ### Neightbor Left
        if neightbour == 'left':
            
            x_min_l = X_center_0 - Width_0 - (a * threshold_x_0)
            x_max_l = X_center_0 - Width_0 + (k * threshold_x_0)

            y_min_l = Y_center_0 - (k * threshold_y_0)
            y_max_l = Y_center_0 + (k * threshold_y_0)

            if (x_min_l < X_center_neightbour < x_max_l and y_min_l < Y_center_neightbour < y_max_l )  :
                list_of_neightbours_l.append([index_bb])
                try:
                    dict_of_neightbours[str(index_it)+'_l'].append(index_bb)
                except KeyError:
                    dict_of_neightbours[str(index_it)+'_l']= []
                    dict_of_neightbours[str(index_it)+'_l'].append(index_bb)
            else:
                list_of_alones_l.append([index_bb])

        ### Neightbor Right
        elif neightbour == 'right':

            x_min_r = X_center_0 + Width_0 - (k*threshold_x_0)
            x_max_r = X_center_0 + Width_0 + (a*threshold_x_0)

            y_min_r = Y_center_0 - (k*threshold_y_0)
            y_max_r = Y_center_0 + (k*threshold_y_0)

            if (x_min_r < X_center_neightbour < x_max_r and y_min_r < Y_center_neightbour < y_max_r ):
                list_of_neightbours_r.append([index_bb])
                try:
                    dict_of_neightbours[str(index_it)+'_r'].append(index_bb)
                except KeyError:
                    dict_of_neightbours[str(index_it)+'_r']= []
                    dict_of_neightbours[str(index_it)+'_r'].append(index_bb)
            else:
                list_of_alones_r.append([index_it, index_bb])
            
    return dict_of_neightbours

def image_mean(x:int, y:int, w:int, h:int, img_path:str) -> float:
    """_summary_

    Args:
        x (int): _description_
        y (int): _description_
        w (int): _description_
        h (int): _description_
        img_path (str): _description_

    Returns:
        float: _description_
    """
    
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = np.mean(gray[y:y + h, x:x + w])

    return roi

def search_voids_bb_neightbours(df_predictions: pd.DataFrame, list_of_dicts: list, h_image:int, w_image:int, img_path) -> list:
    """
    Identifies empty spaces by evaluating each bounding box and it's neightbours. 

    The heuristic of this function draws a "virtual" bounding box beside each bounding box and computes the IoU by iteration over it's neightbours.

    IoU = Intersection over Union (shorturl.at/ituS0)

    If all the calculated IoU (obtained from the neightbours) is less than 10% (0.1), it means that beside the evaluated bounding box, exists an empty space.

    A conditional clause is added with h_image & w_image to avoid surpassing the boundiries of the image in each calculation and avoid irrelevant IoU's.

    Moreover, with image_mean() we double-check the empty spaces. Some empty spaces are not "natural empty spaces", they are just columns or space from each shelf.

    This empty spaces are avoided because do not represent empty *product* spaces.

    Image_mean() returns a floating point value called "roi". Roi is the average pixel value of the a certain space. 

    Args:
        df_predictions (pd.DataFrame): Dataframe with predicted bb obtained with Yolov5.
        list_of_dicts (list): Unified dicts as one list to iterate over. All left and right neightbours added.
        h_image (int): Height of the image.
        w_image (int): Width of the image.
        img_path (_type_): Path to image to be predicted.

    Returns:
        list: List of voids with label and coordinates.
    """
    
    list_of_voids = []

    # Translation of the virtual bounding box. Left = -1 / Right = +1 
    k = 0

    # Void counter
    void_number = 0

    # Iterate over all dicts in list
    for dicts in list_of_dicts:

        # Iterate over key and value pairs of dict (index_a = key // index_b = value pair) - Key = Bounding box being evaluated / Value pair = Neightbours
        for index_a, index_b in dicts.items():

            if index_a[-1] == 'l':
                k = -1
            elif index_a[-1] == 'r':
                k = 1

            # Pick only de integer so the iterator works
            index_a = index_a[0:-2]

            #h_image, w_image = image.shape[0:2] # limits of the image
            w_index_a = df_predictions.loc[int(index_a)][2]
            h_index_a = df_predictions.loc[int(index_a)][3]
            
            # Virtual bounding box to evaluate from neightbours 
            xA1 = df_predictions.loc[int(index_a)][0] - df_predictions.loc[int(index_a)][2]/2 + (k * w_index_a) # Para izquierda: (-1) / Para derecha: 1 / Para arriba: 0
            yA1 = df_predictions.loc[int(index_a)][1] - df_predictions.loc[int(index_a)][3]/2 
            xA2 = df_predictions.loc[int(index_a)][0] + df_predictions.loc[int(index_a)][2]/2 + (k * w_index_a) # Para izquierda: (-1) / Para derecha: 1 / Para arriba: 0
            yA2 = df_predictions.loc[int(index_a)][1] + df_predictions.loc[int(index_a)][3]/2 
            boxA = [xA1, yA1, xA2, yA2]

            X_center_A = df_predictions.loc[int(index_a)][0] - k * df_predictions.loc[int(index_a)][2]  # Left X_center - Width  // Right X_center + Width
            Y_center_A = df_predictions.loc[int(index_a)][1]  - k * df_predictions.loc[int(index_a)][3]  # Left Y_center - Width // Right Y_center + Width

            # Limits of the image
            if 0 < xA1 < w_image and 0 < xA2 < w_image and 0 < yA1 < h_image and  0 < yA2 < h_image:
                    trigger = True

                    # Iterate over each neightbour: neightbour vs virtual bounding box(key value)
                    for item in index_b:
                        
                        first_list = []
                        xB1 = df_predictions.loc[item][0] - df_predictions.loc[item][2]/2
                        yB1 = df_predictions.loc[item][1] - df_predictions.loc[item][3]/2
                        xB2 = df_predictions.loc[item][0] + df_predictions.loc[item][2]/2
                        yB2 = df_predictions.loc[item][1] + df_predictions.loc[item][3]/2
                        boxB = [xB1, yB1,xB2, yB2]

                        xA = max(boxA[0], boxB[0])
                        yA = max(boxA[1], boxB[1])
                        xB = min(boxA[2], boxB[2])
                        yB = min(boxA[3], boxB[3])

                        interArea = (xB - xA) * (yB - yA)

                        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

                        iou = interArea / float(boxAArea + boxBArea - interArea)
                        trigger = trigger and (iou < 0.1)

                    if trigger == False:
                        pass
                    else:
                        roi = image_mean(x= int(xA1),y= int(yA1),w= int(w_index_a), h= int(h_index_a), img_path= img_path)
                        if roi < 50:
                            void_number += 1
                            void_text = f'Void_{void_number}'
                            first_list.append(index_a)
                            first_list.append(void_text)
                            first_list.append(xA1)
                            first_list.append(yA1)
                            first_list.append(xA2)
                            first_list.append(yA2)
                            first_list.append(w_index_a)
                            first_list.append(h_index_a)
                            list_of_voids.append(first_list)
                        else:
                            None

    return list_of_voids

def get_df_voids(list_of_dicts_n:list, df_predictions:pd.DataFrame, h_image:int, w_image:int, img_path:str) -> pd.DataFrame:
    """_summary_

    Args:
        list_of_dicts_n (list): _description_
        df_predictions (pd.DataFrame): _description_
        h_image (int): _description_
        w_image (int): _description_
        img_path (str): _description_

    Returns:
        pd.DataFrame: _description_
    """


    # List of dicts(left/right/high) into 1 list
    list_neightbours = list(itertools.chain.from_iterable(list_of_dicts_n))

    # Get void neightbours 
    list_of_voids = search_voids_bb_neightbours(df_predictions= df_predictions, list_of_dicts=list_neightbours, h_image=h_image, w_image=w_image, img_path= img_path)

    # Create dataframe with data(X_center/Y_center/Label) of voids
    df_voids = pd.DataFrame(list_of_voids, columns=['Neightbour','Label','x1','y1','x2','y2', 'Width','Height'])

    return df_voids

def plot_voids_from_df(img_path:str, df_voids: pd.DataFrame) -> None:
    """_summary_

    Args:
        img_path (str): _description_
        df_voids (pd.DataFrame): _description_
    """

    #image_path_bb = os.path.join(folder_path_final, filename)

    # Refactor this with project paths
    img_path = '/home/cremerf/FinalProject/test_neightbours2.jpg'

    # load the image
    image = cv2.imread(img_path)

    # get the coordinates for each index/rectangle
    for i in df_voids.index:
        x1 = int(df_voids.loc[i][2]) 
        y1 = int(df_voids.loc[i][3]) 
        x2 = int(df_voids.loc[i][4])  
        y2 = int(df_voids.loc[i][5])

        # represents the top left corner of rectangle
        start_point=(x1, y1)

        # represents the top right corner of rectangle
        end_point=(x2,y2)

        # # Blue color in BGR
        color = (0, 0, 255)

        # # Line thickness of 5 px
        thickness = 5

        cv2.putText(image,df_voids.loc[i][1],(x1,y1),cv2.FONT_HERSHEY_PLAIN,0.7,(0,0,0),1)
        # plot the rectangle over the image
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    # Refactor this with project paths
    cv2.imwrite(filename='test_voids1.jpg', img=image)

def run(image_name):
 
    # Instantiate the name of the folder's weights

    training = 'first_training'

    path_training = os.path.join(PATHS.WEIGHTS, training)

    path_training_weights = os.path.join(path_training, 'weights')

    name_of_weights = 'bestnoft.onnx'

    path_picked_weights = os.path.join(path_training_weights, name_of_weights)

    yaml_file = 'config_blmodel.yaml'

    path_yaml = os.path.join(PATHS.DATA, yaml_file)

    yolo = YOLO_Pred(path_picked_weights, path_yaml)

    # Como debe tomar el path para el ml_service?
    img_path = os.path.join(PATHS.UPLOAD_FOLDER, image_name)
    image = cv2.imread(img_path)
    h_image, w_image = image.shape[0:2]
    df_predictions = yolo.predictions(image=image)

    # Get neightbours from 3 ways (right / left / up)
    pool = multiprocessing.Pool()
    neightbour = 'left'
    func = partial(get_neightbours, df_predictions, neightbour)
    dict_of_neightbours_left = pool.map(func, list(df_predictions.index))
    pool.close()
    pool.join()

    # clean up empty positions
    dict_of_neightbours_left = list(filter(None, dict_of_neightbours_left))

    pool = multiprocessing.Pool()
    neightbour = 'right'
    func = partial(get_neightbours, df_predictions, neightbour)
    dict_of_neightbours_right = pool.map(func, list(df_predictions.index))
    pool.close()
    pool.join()

    dict_of_neightbours_right = list(filter(None, dict_of_neightbours_right))

    # Merged 3 separated 
    list_of_dicts_n = [dict_of_neightbours_left, dict_of_neightbours_right]

    # Create dataframe with data(X_center/Y_center/Label) of voids
    df_voids = get_df_voids(list_of_dicts_n= list_of_dicts_n, df_predictions= df_predictions, h_image= h_image, w_image= w_image, img_path=img_path)

    # Plot voids on image (esta hardcodeado el path pero hay que definir donde van a alojarse todas las imagenes con las predicciones)
    plot_voids_from_df(img_path=img_path, df_voids= df_voids)

def main_detect_voids(image_name):
    run(image_name)

if __name__ == '__main__':
    main_detect_voids() 


    


















