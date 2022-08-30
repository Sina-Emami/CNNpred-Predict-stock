from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
    """
    It calculates the number of true positives divided by the number of true positives plus the number
    of false negatives. 
    
    The recall is intuitively the ability of the classifier to find all the positive samples
    
    :param y_true: True labels
    :param y_pred: The predicted values
    :return: Recall is the ratio of the total number of correctly classified positive examples divide to
    the total number of positive examples. High Recall indicates the class is correctly recognized
    (small number of FN).
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
 
def precision_m(y_true, y_pred):
    """
    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number
    of false positives. The precision is intuitively the ability of the classifier not to label as
    positive a sample that is negative.
    
    :param y_true: True labels
    :param y_pred: Predicted labels as floats between 0 and 1
    :return: The precision of the model.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
 
def f1_m(y_true, y_pred):
    """
    The function takes in two arguments, y_true and y_pred, and returns the F1 score
    
    :param y_true: True labels
    :param y_pred: The predicted values
    :return: The F1 score is being returned.
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
 
def f1macro(y_true, y_pred):
    """
    It takes the positive and negative versions of the data and prediction, and averages the F1 scores
    of both
    
    :param y_true: the true labels
    :param y_pred: the predicted values
    :return: The average of the positive and negative versions of the data and prediction.
    """
    f_pos = f1_m(y_true, y_pred)
    # negative version of the data and prediction
    f_neg = f1_m(1-y_true, 1-K.clip(y_pred,0,1))
    return (f_pos + f_neg)/2