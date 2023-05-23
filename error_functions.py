def MSE(Y_pred,Y_true):
      sq_error_sum = 0
      for i in range(len(Y_pred)):
        sq_error_sum += (abs(Y_pred[i]  - Y_true[i]))**2
      return sq_error_sum/len(Y_pred)
    
def mean_abs_error(Y_pred,Y_true):
      error_sum = 0
      for i in range(len(Y_pred)):
        error_sum += (abs(Y_pred[i]  - Y_true[i]))
      return error_sum/len(Y_pred)
    
def L1_regularization(Y_pred,Y_true):
    return mean_abs_error(Y_pred,Y_true)


def L2_regularization(Y_pred,Y_true):
    return MSE(Y_pred,Y_true)


