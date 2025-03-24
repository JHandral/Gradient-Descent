import numpy as np
from numpy.random import randn


def compute_slope_estimator(x_vals, y_vals):
    # Check if the lengths of x and y are the same
    if len(x_vals) != len(y_vals):
        raise ValueError("x and y should have the same length.")
    
    # Compute the mean values
    x_bar = np.mean(x_vals)
    y_bar = np.mean(y_vals)
    
    # Compute the slope using the formula
    numerator = np.sum(x_vals * y_vals) - len(x_vals) * x_bar * y_bar
    denominator = np.sum(x_vals**2) - len(x_vals) * x_bar**2
    
    # Check for division by zero
    if denominator == 0:
        raise ValueError("Denominator is zero, cannot compute slope.")
    
    a = numerator / denominator
    return a


def compute_intercept_estimator(x_vals, y_vals):
    # Check if the lengths of x_vals and y_vals are the same
    if len(x_vals) != len(y_vals):
        raise ValueError("x and y should have the same length.")
    
    # Compute the slope
    a = compute_slope_estimator(x_vals, y_vals)
    
    # Compute the mean values
    x_bar = np.mean(x_vals)
    y_bar = np.mean(y_vals)
    
    # Compute the intercept using the formula
    b = y_bar - a * x_bar
    
    return b



def train_model(x_vals, y_vals):
    # Check if the lengths of x_vals and y_vals are the same
    if len(x_vals) != len(y_vals):
        raise ValueError("x and y should have the same length.")
        
    # Compute optimal slope and intercept values
    a = compute_slope_estimator(x_vals, y_vals)
    b = compute_intercept_estimator(x_vals, y_vals)
    
    return (a, b)



def dL_da(x_vals,y_vals,a,b):
    # Ensure the input arrays have the same length
    if len(x_vals) != len(y_vals):
        raise ValueError("x_vals and y_vals should have the same length.")
    
    n = len(x_vals)
    
    # Calculate the partial derivative of f with respect to a for each data point
    derivatives = -2 * x_vals * (y_vals - a * x_vals - b)
    
    # Return the average of these derivatives
    return np.mean(derivatives)



def dL_db(x_vals,y_vals,a,b):
    # Ensure the input arrays have the same length
    if (len(x_vals) != len(y_vals)):
        raise ValueError("x_vals and y_vals should have the same length.")
    
    n = len(x_vals)
    
    # Calculate the partial derivative of f with respect to b for each data point
    derivatives = -2 * (y_vals - a * x_vals - b)
    
    # Return the average of these derivatives
    return np.mean(derivatives)



def gradient_descent_step(x_vals,y_vals,a,b,k=0.01):
    # Calculate the partial derivatives
    da = dL_da(x_vals, y_vals, a, b)
    db = dL_db(x_vals, y_vals, a, b)
    
    # Update the values of a and b
    a_updated = a - (k/len(x_vals)) * da
    b_updated = b - (k/len(x_vals)) * db
    
    return (a_updated, b_updated)



def gradient_descent(x_vals,y_vals,a_0=0,b_0=0,k=1000):
    a_current, b_current = a_0, b_0
    for _ in range(k):
        a_current, b_current = gradient_descent_step(x_vals, y_vals, a_current, b_current)
    return a_current, b_current




def einsum_1(A, B):
	return np.einsum('ij,ij->ij', A, B)




def einsum_2(A, B):
	return np.einsum('ij,j->ij', A, B)




#gpt4
def einsum_3(A, B):
	return np.einsum('ijk,ik->ij', A, B)



#gpt4
def einsum_4(A, B):
	return np.einsum('ijk,ikl->ijl', A, B)