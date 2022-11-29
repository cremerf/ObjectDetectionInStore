from packages.yolo_predict import YOLO_Pred
import pandas as pd
import cv2


def get_neightbours(df_predictions:pd.DataFrame) -> list:

    dict_l_n = {} # neigh left
    dict_r_n = {} # neigh right
    dict_h_n= {}  # alone right

    for index_bb in df_predictions.index:
        list_of_neightbours_l = []
        list_of_neightbours_r = []
        list_of_neightbours_h = [] 
        list_of_alones_l = []
        list_of_alones_r = []
        list_of_alones_h=[]
        for index in df_predictions.index:
            X_center_0 = df_predictions.loc[index_bb][0]
            Y_center_0 = df_predictions.loc[index_bb][1]
            Width_0 = df_predictions.loc[index_bb][2]
            height_0 = df_predictions.loc[index_bb][3]
            threshold_x_0 = 0.8 * Width_0
            threshold_y_0 = 0.8 * height_0

            X_center_neightbour = df_predictions.loc[index][0]
            Y_center_neightbour = df_predictions.loc[index][1]
            a=4
            k=1.2

            ### Neightbor Left
            x_min_l = X_center_0 - Width_0 - (a*threshold_x_0)
            x_max_l = X_center_0 - Width_0 + (k*threshold_x_0)

            y_min_l = Y_center_0 - (k*threshold_y_0)
            y_max_l = Y_center_0 + (k*threshold_y_0)
            
            ### Neightbor Right
            x_min_r = X_center_0 + Width_0 - (k*threshold_x_0)
            x_max_r = X_center_0 + Width_0 + (a*threshold_x_0)

            y_min_r = Y_center_0 - (k*threshold_y_0)
            y_max_r = Y_center_0 + (k*threshold_y_0)
            
            ### Neightbor High
            x_min_h = X_center_0 - (k*threshold_x_0)
            x_max_h = X_center_0 + (2*k*threshold_x_0)

            y_min_h = Y_center_0 + height_0 - (0.8*threshold_y_0)
            y_max_h = Y_center_0 + height_0 + (2*a*threshold_y_0)
            
            ### Search neig left
            if (x_min_l < X_center_neightbour < x_max_l and y_min_l < Y_center_neightbour < y_max_l )  :
                list_of_neightbours_l.append([index])
                try:
                    dict_l_n[str(index_bb)].append(index)
                except KeyError:
                    dict_l_n[str(index_bb)]=[]
                    dict_l_n[str(index_bb)].append(index)
                
            else:
                list_of_alones_l.append([index])
            if (x_min_r < X_center_neightbour < x_max_r and y_min_r < Y_center_neightbour < y_max_r ):
                list_of_neightbours_r.append([index])
                try:
                    dict_r_n[str(index_bb)].append(index)
                except KeyError:
                    dict_r_n[str(index_bb)]=[]
                    dict_r_n[str(index_bb)].append(index)
            else:
                list_of_alones_r.append([index_bb, index])
                
            # Search neig hight    
            if (x_min_h < X_center_neightbour < x_max_h and y_min_h < Y_center_neightbour < y_max_h):
                list_of_neightbours_h.append([index])
                try:
                    dict_h_n[str(index_bb)].append(index)
                except KeyError:
                    dict_h_n[str(index_bb)]=[]
                    dict_h_n[str(index_bb)].append(index)
            else:
                list_of_alones_h.append([index_bb, index])

    return [dict_l_n, dict_r_n, dict_h_n]

def search_voids_bb_neightbours(df_predictions: pd.DataFrame, img_path:str, dict_of_neightbours: dict,  neightbour:str=('left','right','up')):
    
    list_of_voids = []

    k = 0
    z = 0
    if neightbour == 'left':
        k = -1
        z = 0
    elif neightbour == 'right':
        k = 1
        z = 0
    else:
        k = 0
        z = -1
    
    image = cv2.imread(filename= img_path)

    for index_a, index_b in dict_of_neightbours.items():

        h_image, w_image = image.shape[0:2] # obtengo limites de la imagen
        w_index_a = df_predictions.loc[int(index_a)][2]
        h_index_a = df_predictions.loc[int(index_a)][3]


        # Virtual bounding box to evaluate neightbours 
        xA1 = df_predictions.loc[int(index_a)][0] - df_predictions.loc[int(index_a)][2]/2 + (k * w_index_a) # Para izquierda: (-1) / Para derecha: 1 / Para arriba: 0
        yA1 = df_predictions.loc[int(index_a)][1] - df_predictions.loc[int(index_a)][3]/2 + (z * h_index_a) # Para izquierda: 0 / Para derecha: 0 / Para arriba: (-1)
        xA2 = df_predictions.loc[int(index_a)][0] + df_predictions.loc[int(index_a)][2]/2 + (k * w_index_a) # Para izquierda: (-1) / Para derecha: 1 / Para arriba: 0
        yA2 = df_predictions.loc[int(index_a)][1] + df_predictions.loc[int(index_a)][3]/2 + (z * h_index_a) # Para izquierda: 0 / Para derecha: 0 / Para arriba: -1
        boxA = [xA1, yA1, xA2, yA2]

        X_center_A = df_predictions.loc[int(index_a)][0] - k * df_predictions.loc[int(index_a)][2]  # Left X_center - Width  // Right X_center + Width
        Y_center_A = df_predictions.loc[int(index_a)][1]  - k * df_predictions.loc[int(index_a)][2]  # Left Y_center - Width // Right Y_center + Width

        print(f'Evaluating {index_a}...')

        # Limits of the image
        if 0 < xA1 < w_image and 0 < xA2 < w_image and 0 < yA1 < h_image and  0 < yA2 < h_image:
                trigger = True
                void_number = 0

                # Iterate over each neightbour: neightbour vs virtual bounding box (index_a = which we are evaluating)
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
                    void_text = f'{neightbour} void #{void_number}'
                    first_list.append(index_a)
                    first_list.append(void_text)
                    first_list.append(X_center_A)
                    first_list.append(Y_center_A)
                    list_of_voids.append(first_list)

    return list_of_voids

def df_voids():
    pass

def search_voids_no_neightbours():
    pass


def run():

    #crear modulo con paths para weights + yaml 

    # Load model & YAML file
    yolo = YOLO_Pred('/home/cremerf/FinalProject/data/first_training/weights/bestnoft.onnx', '/home/cremerf/FinalProject/data/config_blmodel.yaml')

    # Como debe tomar el path para el ml_service?
    img_path = '/home/cremerf/FinalProject/eudes-fede/test_imgs/test_7.jpg'
    df_predictions = yolo.predictions(img_path=img_path)

    # Get neightbours from 3 ways (right / left / up)
    dict_l_n

    


















