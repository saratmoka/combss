import numpy as np
from sklearn import metrics


def performance_metrics(data_X, beta_true, beta_pred):
    """
    Function for computing performance metrics.
    """
  
    s_true = [beta_true != 0][0]
    s_pred = [beta_pred != 0][0]
    c_matrix = metrics.confusion_matrix(s_true, s_pred)
    
    TN = c_matrix[0, 0]
    FN = c_matrix[1, 0]
    FP = c_matrix[0, 1]
    TP = c_matrix[1, 1]
    
    
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    sensitivity = TP/(TP + FN)
    specificity = TN/(TN + FP)
    
    
    if (sum(s_pred) == 0):
        pe = 1
        precision = 'NA'
        MCC = 'NA'
    else:
        Xbeta_true = data_X@beta_true
        pe = np.square(Xbeta_true - data_X@beta_pred).mean()/np.square(Xbeta_true).mean()        
        precision =  TP/(TP + FP)
        # print('Pre + Sens:', precision + sensitivity)
        MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if TP == 0:
        f1_score = 0.0
    else:
        f1_score = TP/(TP + (FP + FN)/2)
        #f1_score = (2*precision*sensitivity)/(precision + sensitivity)
    
    # accuracy = round(accuracy, 4)
    # sensitivity = round(sensitivity, 4)
    # specificity = round(specificity, 4)
    # pe = round(pe, 4)
    # precision = round(precision, 4)
    # f1_score = round(f1_score, 4)
    # MCC = round(MCC, 4)
    
    #print(pe, MCC, accuracy, sensitivity, specificity, f1_score, precision)
        
    return [pe, MCC, accuracy, sensitivity, specificity, f1_score, precision]
