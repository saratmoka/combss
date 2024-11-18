"""
combss._metrics.py

This private module contains logic for computing performance metrics for variable selection.
Metrics computed:
- Relative predition error
- Matthew's Correlation Coefficient 
- Accuracy
- Sensitivity
- Specificity
- F1 Score
- Precision
"""

import numpy as np
from sklearn import metrics



def performance_metrics(data_X, beta_true, beta_pred):
    
    """ Computes the evaluation metrics for COMBSS.

	Parameters
	----------
	data_X : array-like of shape (n_samples, n_covariates)
        The design matrix, where `n_samples` is the number of samples observed
        and `n_covariates` is the number of covariates measured in each sample.

    beta_true : array-like of shape (n_covariates, 1)
        The true value of beta used in the generation of data.

    beta_pred : array-like of shape (n_covariates, 1)
        The predicted value of beta generated by COMBSS.


	Returns
	-------
    array-like of floats, [pe, MCC, accuracy, sensitivity, specificity, f1_score, precision], where

	pe : float
        The model's relative prediction error, expressed as a fraction where the L-2 norm of the difference 
        between the fitted values and true predicted values is divided by the L-2 norm of the true predicted values.

    MCC : float
        The model's Matthew's Correlation Coefficient.

    acc : float
        The accuracy of the particular model, calculated as proportion of total instances where 
        the model correctly classifies whether or not a predictor is selected in, or rejected from 
        the true model, calculated as a quantity between 0 and 1.

    sens : float
        The sensitivity of the particular model, calculated as the proportion of total instances
        where the model correctly classifies the inclusion of predictors that belong in the true
        model, calculated as a quantity between 0 and 1.

    spec : float
        The specificity of the particular model, calculated as the proportion of total instances 
        where the model correctly classifies the rejection of predictors that do not belong in 
        the true model, calculated as a quantity between 0 and 1.

    f1 : float
        The F1 Score of the particular model.

    prec : float
        The precision of the model, calculated as the proportion at which the model correctly 
        includes a true predictor in it's predicted model, calculated as a quantity between 0 and 1.
    """
  
    s_true = [beta_true != 0][0]
    s_pred = [beta_pred != 0][0]
    c_matrix = metrics.confusion_matrix(s_true, s_pred)
    
    TN = c_matrix[0, 0]
    FN = c_matrix[1, 0]
    FP = c_matrix[0, 1]
    TP = c_matrix[1, 1]
    
    acc = (TP + TN)/(TP + TN + FP + FN)
    sens = TP/(TP + FN)
    spec = TN/(TN + FP)
    
    # If the model fails to predict any elements within the predicted model, we take Prediction Error = 1, Precision = 0 and MCC = 0.
    if (sum(s_pred) == 0):
        pe = 1
        prec = 0
        mcc = 0
    else:
        Xbeta_true = data_X@beta_true
        pe = np.square(Xbeta_true - data_X@beta_pred).mean()/np.square(Xbeta_true).mean()        
        prec =  TP/(TP + FP)
        mcc = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if TP == 0:
        # If the model fails to recover any existing elements of the true model, we take F1 Score = 0.
        f1 = 0.0
    else:
        f1 = TP/(TP + (FP + FN)/2)
    
    result = {
        "pe" : pe,
        "mcc" : mcc,
        "acc" : acc,
        "sens" : sens,
        "spec" : spec,
        "f1" : f1,
        "prec" : prec
        }
        
    return result
