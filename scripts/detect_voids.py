from packages.yolo_predict import YOLO_Pred
import pandas as pd

#crear modulo con paths para weights + yaml 
yolo = YOLO_Pred('/home/cremerf/FinalProject/data/first_training/weights/bestnoft.onnx', '/home/cremerf/FinalProject/data/config_blmodel.yaml')

def get_neightbours(img_path:str):

    df_predictions = yolo.predictions(img_path=img_path)

    dict_l_n = {  } # neigh left
    dict_r_n = { } # neigh right
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
                None
                if index_bb==694 and index==693:
                    print(f'El bounding box {index_bb} tiene al bb {index} de vecino')
                    print(x_min_l, X_center_neightbour, x_max_l, "___", y_min_l, Y_center_neightbour,y_max_l )
                list_of_neightbours_l.append([index])
                try:
                    dict_l_n[str(index_bb)].append(index)
                except KeyError:
                    dict_l_n[str(index_bb)]=[]
                    dict_l_n[str(index_bb)].append(index)
                
            else:
                if index_bb==288 and index==277:
                    print(f'El bounding box {index_bb} no tiene vencidad con {index}')
                    print(x_min_l, X_center_neightbour, x_max_l, "___", y_min_l, Y_center_neightbour,y_max_l )
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
            if (x_min_h < X_center_neightbour < x_max_h and y_min_h < Y_center_neightbour < y_max_h ):
                list_of_neightbours_h.append([index])
                try:
                    dict_h_n[str(index_bb)].append(index)
                except KeyError:
                    dict_h_n[str(index_bb)]=[]
                    dict_h_n[str(index_bb)].append(index)
            else:
                list_of_alones_h.append([index_bb, index])

    return [dict_l_n, dict_r_n, dict_h_n]


def bb_intersection_over_union(df_predictions: pd.DataFrame, index_a:int, index_b:int) -> float:

    a = index_a
    xA1 = df_predictions.loc[a][0]-df_predictions.loc[a][2]/2
    yA1 = df_predictions.loc[a][1]+df_predictions.loc[a][3]/2
    xA2 = df_predictions.loc[a][0]+df_predictions.loc[a][2]/2
    yA2 = df_predictions.loc[a][1]-df_predictions.loc[a][3]/2
    boxA = [xA1, yA1, xA2, yA2 ]

    b = index_b # indices vecinos detectados que estan en el diccionario
    xB1 = df_predictions.loc[b][0]-df_predictions.loc[b][2]/2
    yB1 = df_predictions.loc[b][1]+df_predictions.loc[b][3]/2
    xB2 = df_predictions.loc[b][0]+df_predictions.loc[b][2]/2
    yB2 = df_predictions.loc[b][1]-df_predictions.loc[b][3]/2
    boxB = [xB1, yB1,xB2, yB2 ]
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])


    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def search_empty_bounding_box(df_predictions: pd.DataFrame, dict_of_neightbours: dict):









