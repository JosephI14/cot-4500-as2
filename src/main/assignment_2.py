import numpy as np

# Question 1
def Neville_one(x, y, z):

  n = len(x)
  
  matrix = np.zeros([n,n], dtype=float)

  for i in range(n):
    
    matrix[i][0] = y[i]

  for i in range(1, n):
    for j in range(1, n):
      
      term1 = (z - x[i-j]) * matrix[i][j-1]
      
      term2 = (z-x[i]) * matrix[i- 1][j-1]

      matrix[i][j] = (term1 - term2) / (x[i] - x[i-j])
  
  print(matrix[2][2], end="\n\n")

# Question 2
def Newton_two(x, y):
  
  n = len(x)

  matrix = np.zeros([n,n], dtype=float)

  for i in range(n):
    
    matrix[i][0] = y[i]
    
  for i in range(1, n):
    
    for j in range(1, n):

      matrix[i][j] = (matrix[i][j-1] - matrix[i-1][j-1])/(x[i]-x[i-j])

  return matrix
  
# Question 3
def newton_ford_approx(matrix, x, y, appx_val):

  
  deg1 = y[0] + (matrix[1][1] * (appx_val - x[0]))

  
  deg2 = deg1 + (matrix[2][2] * ((appx_val - x[0]) * (appx_val - x[1])))

  
  deg3 = deg2 + (matrix[3][3] * ((appx_val - x[0]) * (appx_val - x[1]) * (appx_val - x[2])))

  return deg3
  
  
  
# Question 4
def divided_difference(matrix: np.array):

  # Divided difference math is below
    size = len(matrix)
  
    for i in range(2, size):
        for j in range(2, i + 2):
            if j >= len(matrix[i]) or matrix[i][j] != 0:
                continue

            
            lc: float = matrix[i][j - 1]
            
            dlc: float = matrix[i - 1][j - 1]
            
            num: float = (lc - dlc)
            
            den = matrix[i][0] - matrix[i - j + 1][0]
            
            ops = num / den
          
            matrix[i][j] = ops

    return matrix


def hermite_poly():
  
    herm_x = [3.6, 3.8, 3.9]
  
    herm_fx = [1.675, 1.436, 1.318]
  
    herm_prime = [-1.195, -1.188, -1.182]
    
    m = len(herm_x)
  
    matrix = np.zeros((2 * m, 2 * m))
    
    for i, x in enumerate(herm_x):
      
        matrix[2 * i][0] = x
      
        matrix[2 * i + 1][0] = x

    
    for i, y in enumerate(herm_fx):
      
        matrix[2 * i][1] = y
      
        matrix[2 * i + 1][1] = y

    
    for i, f_prime in enumerate(herm_prime):
      
        matrix[2 * i + 1][2] = f_prime

    finalQ4 = divided_difference(matrix)
    print(finalQ4, end="\n\n")

# Question 5
def cubic_spline_interpolation():
  # I wasn't able to figure out this one quite
  cubic_x = [2, 5, 8, 10]
  
  cubic_fx = [3, 5, 7, 9]
  
  m = len(cubic_x)
  
  cub1 = np.zeros(m - 1)
  
  cub2 = np.zeros(m)
  
  for i in range(m - 1):
      cub1[i] = cubic_x[i + 1] - cubic_x[i]
    
      cub2[i + 1] = (cubic_fx[i + 1] - cubic_fx[i]) / cub1[i]

  mat_A = np.zeros((m, m))
  
  vector_b = np.zeros(m)
  
  for i in range(1, m - 1):
    
      mat_A[i][i - 1] = cub1[i - 1]
    
      mat_A[i][i] = 2 * (cub1[i - 1] + cub1[i])
    
      mat_A[i][i + 1] = cub1[i]
    
      vector_b[i] = 3 * (cub2[i + 1] - cub2[i])

  mat_A[0][0] = 1
  
  mat_A[m - 1][m - 1] = 1
# Creating vector x
  vector_x = np.linalg.solve(mat_A, vector_b)

# Printing Matrix A, which is the first part of the question
  print(mat_A, end="\n\n")
  
# This is printing vector b
  print(vector_b, end="\n\n")
  
# This is printing vector x
  print(vector_x, end="\n\n")

# I will call the cubic_spline_interpolation command from main
  
  
  
if __name__ == "__main__":
  np.set_printoptions(precision=7, suppress=True, linewidth=100)
# Q1 main 
  nev_x = np.asarray([3.6, 3.8, 3.9])
  nev_fx = np.asarray([1.675, 1.436, 1.318])
  nev_approx = 3.7

  Neville = Neville_one(nev_x, nev_fx, nev_approx)
# Q2 main
  newt_x = np.asarray([7.2, 7.4, 7.5, 7.6])
  newt_fx = np.asarray([23.5492, 25.3913, 26.8224, 27.4589])

  answer_two = Newton_two(newt_x, newt_fx)
 
  print("[", end="")
  print(answer_two[1][1], end=", ")
  print(answer_two[2][2], end=", ")
  print(answer_two[3][3], end="]\n\n")

# Q3 main
  print(newton_ford_approx(answer_two, newt_x, newt_fx, 7.3), end="\n\n")

# Q4 main
  hermite_poly()
  
# Q5 main
  cubic_spline_interpolation()

