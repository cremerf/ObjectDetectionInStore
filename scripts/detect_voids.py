from packages.yolo_predict import YOLO_Pred
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import cv2
import multiprocessing
from functools import partial
import itertools


def get_neightbours(df_predictions:pd.DataFrame, neightbour:str, index_it:int) -> dict:
    """_summary_

    Args:
        df_predictions (pd.DataFrame): _description_
        neightbour (str): _description_
        index_it (int): _description_

    Returns:
        dict: _description_
    """

    dict_of_neightbours = {} 

    X_center_0 = df_predictions.loc[index_it][0]
    Y_center_0 = df_predictions.loc[index_it][1]
    Width_0 = df_predictions.loc[index_it][2]
    height_0 = df_predictions.loc[index_it][3]
    threshold_x_0 = 0.8 * Width_0
    threshold_y_0 = 0.8 * height_0

    INPUT_WH_YOLO = 640
    filter_1 = (df_predictions.loc[:, 'X_center'] < (X_center_0 + INPUT_WH_YOLO/8))
    filter_2 = (df_predictions.loc[:, 'X_center'] > (X_center_0 - INPUT_WH_YOLO/8))
    filter_3 = (df_predictions.loc[:, 'Y_center'] > (Y_center_0 - INPUT_WH_YOLO/8))
    filter_4 = (df_predictions.loc[:, 'Y_center'] < (Y_center_0 + INPUT_WH_YOLO/8))
    filter_final = (filter_1 & filter_2) & (filter_3 & filter_4)

    df_predictions_final_2 = df_predictions[filter_final]

    for index_bb in df_predictions_final_2.index:

        list_of_neightbours_l = []
        list_of_neightbours_r = []
        list_of_neightbours_h = [] 
        list_of_alones_l = []
        list_of_alones_r = []
        list_of_alones_h=[]

        X_center_neightbour = df_predictions.loc[index_bb][0]
        Y_center_neightbour = df_predictions.loc[index_bb][1]
        a = 4
        k = 1.2

        ### Neightbor Left
        if neightbour == 'left':
            
            x_min_l = X_center_0 - Width_0 - (a*threshold_x_0)
            x_max_l = X_center_0 - Width_0 + (k*threshold_x_0)

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

        ### Neightbor High
        else:
            x_min_h = X_center_0 - (k*threshold_x_0)
            x_max_h = X_center_0 + (2*k*threshold_x_0)

            y_min_h = Y_center_0 + height_0 - (0.8*threshold_y_0)
            y_max_h = Y_center_0 + height_0 + (2*a*threshold_y_0)

            if (x_min_h < X_center_neightbour < x_max_h and y_min_h < Y_center_neightbour < y_max_h ):
                list_of_neightbours_h.append([index_bb])
                try:
                    dict_of_neightbours[str(index_it)+'_h'].append(index_bb)
                except KeyError:
                    dict_of_neightbours[str(index_it)+'_h']= []
                    dict_of_neightbours[str(index_it)+'_h'].append(index_bb)
            else:
                list_of_alones_h.append([index_it, index_bb])

    return dict_of_neightbours

def search_voids_bb_neightbours(df_predictions: pd.DataFrame, list_of_dicts: list, h_image:int, w_image:int):
    """_summary_

    Args:
        df_predictions (pd.DataFrame): _description_
        list_of_dicts (list): _description_
        h_image (int): _description_
        w_image (int): _description_

    Returns:
        _type_: _description_
    """
    
    list_of_voids = []

    k = 0
    z = 0
    
    # iterate over all dicts in list
    for dicts in list_of_dicts:

        # iterate over key and value pairs of dict (index_a = key // index_b = value pair)
        for index_a, index_b in dicts.items():

            if index_a[-1] == 'l':
                k = -1
                z = 0
            elif index_a[-1] == 'r':
                k = 1
                z = 0
            else:
                k = 0
                z = -1

            index_a = index_a[0:-2]

            #h_image, w_image = image.shape[0:2] # limits of the image
            w_index_a = df_predictions.loc[int(index_a)][2]
            h_index_a = df_predictions.loc[int(index_a)][3]

            
            # Virtual bounding box to evaluate neightbours 
            xA1 = df_predictions.loc[int(index_a)][0] - df_predictions.loc[int(index_a)][2]/2 + (k * w_index_a) # Para izquierda: (-1) / Para derecha: 1 / Para arriba: 0
            yA1 = df_predictions.loc[int(index_a)][1] - df_predictions.loc[int(index_a)][3]/2 + (z * h_index_a) # Para izquierda: 0 / Para derecha: 0 / Para arriba: (-1)
            xA2 = df_predictions.loc[int(index_a)][0] + df_predictions.loc[int(index_a)][2]/2 + (k * w_index_a) # Para izquierda: (-1) / Para derecha: 1 / Para arriba: 0
            yA2 = df_predictions.loc[int(index_a)][1] + df_predictions.loc[int(index_a)][3]/2 + (z * h_index_a) # Para izquierda: 0 / Para derecha: 0 / Para arriba: -1
            boxA = [xA1, yA1, xA2, yA2]


            X_center_A = df_predictions.loc[int(index_a)][0] - k * df_predictions.loc[int(index_a)][2]  # Left X_center - Width  // Right X_center + Width
            Y_center_A = df_predictions.loc[int(index_a)][1]  - k * df_predictions.loc[int(index_a)][3]  # Left Y_center - Width // Right Y_center + Width

            print(f'Evaluating {index_a}...')
            

            # Limits of the image
            if 0 < xA1 < w_image and 0 < xA2 < w_image and 0 < yA1 < h_image and  0 < yA2 < h_image:
                    trigger = True
                    void_number = 0

                    # Iterate over each neightbour: neightbour vs virtual bounding box
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
                    #print(f'Trigger: {trigger}')

                    if trigger == False:
                        pass
                        #print(f'No hay espacio vacio a la izquierda de {index_a}')
                    else:
                        print(f'A la izquierda de {index_a} hay espacio vacio')
                        void_number += 1
                        void_text = f'{index_a} void #{void_number}'
                        first_list.append(index_a)
                        first_list.append(void_text)
                        first_list.append(xA1)
                        first_list.append(yA1)
                        first_list.append(xA2)
                        first_list.append(yA2)
                        first_list.append(w_index_a)
                        first_list.append(h_index_a)
                        list_of_voids.append(first_list)
                    

    return list_of_voids

def df_voids():
    pass

def plot_voids_from_df():
    pass


def run():

    #crear modulo con paths para weights + yaml 

    # Load model & YAML file
    yolo = YOLO_Pred('/home/cremerf/FinalProject/data/first_training/weights/bestnoft.onnx', '/home/cremerf/FinalProject/data/config_blmodel.yaml')

    # Como debe tomar el path para el ml_service?
    img_path = '/home/cremerf/FinalProject/eudes-fede/test_imgs/test_7.jpg'
    image = cv2.imread(img_path)
    h_image, w_image = image.shape[0:2] # limits of the image ## OJO, ME PARECE QUE EL PRIMER VALOR ES EL WIDTH, NO EL HEIGHT
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

    pool = multiprocessing.Pool()
    neightbour = 'up'
    func = partial(get_neightbours, df_predictions, neightbour)
    dict_of_neightbours_up = pool.map(func, list(df_predictions.index))
    pool.close()
    pool.join()

    dict_of_neightbours_up = list(filter(None, dict_of_neightbours_up))

    # Merged 3 separated list of dicts(left/right/high) into 1 list
    dicts_neightbours = [dict_of_neightbours_left, dict_of_neightbours_right, dict_of_neightbours_up]
    list_neightbours = list(itertools.chain.from_iterable(dicts_neightbours))

    # Get void neightbours 
    list_of_voids = search_voids_bb_neightbours(df_predictions=df_predictions, list_of_dicts=list_neightbours, h_image=h_image, w_image=w_image)

    # Create dataframe with data(X_center/Y_center/Label) of voids
    df_voids = pd.DataFrame(list_of_voids, columns=['Neightbour','Label','X_center','Y_center','Width','Height'])

    # Plot voids on image



    


















