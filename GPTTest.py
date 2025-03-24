import numpy as np
from numpy.random import randn




#gpt4
def compute_slope_estimator(x_vals,y_vals):
    # Check if the lengths of x and y are the same
    if (len(x_vals) != len(y_vals)):
        raise ValueError("x and y should have the same length.")
    
    # Compute the mean values
    x_bar = np.mean(x_vals)
    y_bar = np.mean(y_vals)
    
    # Compute the slope using the formula
    numerator = np.sum((x_vals * y_vals) - len(x_vals) * x_bar * y_bar)
    denominator = np.sum((x_vals**2) - len(x_vals) * x_bar**2)
    
    # Check for division by zero
    if denominator == 0:
        raise ValueError("Denominator is zero, cannot compute slope.")
    
    a = numerator / denominator
    
    return a
#gpt3.5
import numpy as np

def compute_slope_estimator(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum(x * y) - n * x_mean * y_mean
    denominator = np.sum(x**2) - n * x_mean**2

    slope_estimator = numerator / denominator

    return slope_estimator
# correct
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
# In gpt's code, when computing the sums, it's multiplying the entire arrays, 
# which means that it's performing element-wise multiplication. 
# This is correct, but when subtracting, should be using the aggregated sum, not multiplying the length again.  




#gpt4
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
#gpt3.5
def compute_intercept_estimator(x, y):
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope_estimator = compute_slope_estimator(x, y)  # Reuse the previously defined function

    intercept_estimator = y_mean - slope_estimator * x_mean

    return intercept_estimator




#gpt4
def train_model(x_vals, y_vals):
    # Check if the lengths of x_vals and y_vals are the same
    if len(x_vals) != len(y_vals):
        raise ValueError("x and y should have the same length.")
        
    # Compute optimal slope and intercept values
    a = compute_slope_estimator(x_vals, y_vals)
    b = compute_intercept_estimator(x_vals, y_vals)
    
    return (a, b)

#gpt3.5
def train_model(x, y):
    slope = compute_slope_estimator(x, y)
    intercept = compute_intercept_estimator(x, y)
    return slope, intercept



#gpt4 
def dL_da(x_vals,y_vals,a,b):
    # Ensure the input arrays have the same length
    if len(x_vals) != len(y_vals):
        raise ValueError("x_vals and y_vals should have the same length.")
    
    n = len(x_vals)
    
    # Calculate the partial derivative of f with respect to a for each data point
    derivatives = -2 * x_vals * (y_vals - a * x_vals - b)
    
    # Return the average of these derivatives
    return np.mean(derivatives)
#gpt3.5
def dL_da(x_vals, y_vals, a, b):
    n = len(x_vals)
    sum_derivative = 0
    
    for i in range(n):
        # Compute the derivative of f(x_i, y_i, a, b) with respect to 'a'
        derivative = -2 * (y_vals[i] - a * x_vals[i] - b) * x_vals[i]
        sum_derivative += derivative
    
    partial_derivative = (1/n) * sum_derivative
    return partial_derivative


#gpt4
def dL_db(x_vals,y_vals,a,b):
    # Ensure the input arrays have the same length
    if (len(x_vals) != len(y_vals)):
        raise ValueError("x_vals and y_vals should have the same length.")
    
    n = len(x_vals)
    
    # Calculate the partial derivative of f with respect to b for each data point
    derivatives = -2 * (y_vals - a * x_vals - b)
    
    # Return the average of these derivatives
    return np.mean(derivatives)
#gpt3.5
def dL_db(x_vals, y_vals, a, b):
    n = len(x_vals)
    sum_derivative = 0
    
    for i in range(n):
        # Compute the derivative of f(x_i, y_i, a, b) with respect to 'b'
        derivative = -2 * (y_vals[i] - a * x_vals[i] - b)
        sum_derivative += derivative
    
    partial_derivative = (1/n) * sum_derivative
    return partial_derivative


#gpt4
def gradient_descent_step(x_vals,y_vals,a,b,k=0.01):
    # Calculate the partial derivatives
    da = dL_da(x_vals, y_vals, a, b)
    db = dL_db(x_vals, y_vals, a, b)
    
    # Update the values of a and b
    a_updated = a - (k/len(x_vals)) * da
    b_updated = b - (k/len(x_vals)) * db
    
    return (a_updated, b_updated)
#gpt3.5
def gradient_descent_step(x_vals, y_vals, a, b, k=1000):
    n = len(x_vals)
    # Calculate the partial derivatives using the functions from Problems 4 and 5
    partial_derivative_a = dL_da(x_vals, y_vals, a, b)
    partial_derivative_b = dL_db(x_vals, y_vals, a, b)

    # Update the values of 'a' and 'b' using the gradient descent formula
    a_updated = a - (k/n) * partial_derivative_a
    b_updated = b - (k/n) * partial_derivative_b

    return a_updated, b_updated


#gpt4
def gradient_descent(x_vals,y_vals,a_0=0,b_0=0,k=1000):
    a_current, b_current = a_0, b_0
    for _ in range(k):
        a_current, b_current = gradient_descent_step(x_vals, y_vals, a_current, b_current)
    return a_current, b_current
#gpt3.5
def gradient_descent(x_vals, y_vals, a_0, b_0, k=1):
    a_current = a_0
    b_current = b_0
    for _ in range(k):
        a_current, b_current = gradient_descent_step(x_vals, y_vals, a_current, b_current)
    return a_current, b_current


#gpt4
def einsum_1(A, B):
	return np.einsum('ij,ij->ij', A, B)
#gpt3.5
def einsum_1(A, B):
    C = np.einsum('ij,ij->ij', A, B)
    return C


#gpt4
def einsum_2(A, B):
	return np.einsum('ij,j->ij', A, B)
#gpt3.5
def einsum_2(A, B):
    C = np.einsum('ij,j->ij', A, B)
    return C


#gpt4
def einsum_3(A, B):
	return np.einsum('ijk,ik->ij', A, B)
#gpt3.5
def einsum_3(A, B):
    # Ensure that the shapes of A and B are compatible for multiplication
    if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:
        raise ValueError("Incompatible shapes for batch matrix-vector multiplication.")

    # Perform the batch matrix-vector multiplication
    C = np.einsum('ijk,ik->ij', A, B)

    return C


#gpt4
def einsum_4(A, B):
	return np.einsum('ijk,ikl->ijl', A, B)
#gpt3.5
def einsum_4(A, B):
    # Ensure that the shapes of A and B are compatible for multiplication
    if A.shape[0] != B.shape[0] or A.shape[2] != B.shape[1]:
        raise ValueError("Incompatible shapes for batch matrix-matrix multiplication.")

    # Perform the batch matrix-matrix multiplication
    C = np.einsum('ijk,ikq->ijq', A, B)

    return C
