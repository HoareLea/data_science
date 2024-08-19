import numpy as np

# Creating an array from a list
arr1 = np.array([1, 2, 3, 4, 5])
print("Array created from a list:\n", arr1)

# Creating a multi-dimensional array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("\nMulti-dimensional array:\n", arr2)

# Creating arrays using numpy functions
arr3 = np.zeros((2, 3))  # 2x3 array of zeros
print("\nArray of zeros:\n", arr3)

arr4 = np.ones((3, 2))  # 3x2 array of ones
print("\nArray of ones:\n", arr4)

arr5 = np.eye(3)  # 3x3 identity matrix
print("\nIdentity matrix:\n", arr5)

arr6 = np.arange(10)  # Array with values from 0 to 9
print("\nArray with values from 0 to 9:\n", arr6)

arr7 = np.linspace(0, 1, 5)  # 5 values from 0 to 1 inclusive
print("\nArray with 5 values from 0 to 1:\n", arr7)

# Reshaping an array
arr8 = np.arange(12).reshape(3, 4)  # Reshape to 3x4 array
print("\nReshaped array to 3x4:\n", arr8)

# Accessing elements
print("\nElement at position (1, 2) in arr8:\n", arr8[1, 2])

# Slicing arrays
print("\nSlicing arr8 to get the first two rows and columns 1 and 2:\n", arr8[:2, 1:3])

# Boolean indexing
arr9 = np.array([10, 20, 30, 40, 50])
print("\nElements in arr9 greater than 25:\n", arr9[arr9 > 25])

# Arithmetic operations
arr10 = np.array([1, 2, 3])
arr11 = np.array([4, 5, 6])
print("\nAdding arr10 and arr11:\n", arr10 + arr11)
print("\nMultiplying arr10 by 2:\n", arr10 * 2)

# Statistical operations
arr12 = np.array([1, 2, 3, 4, 5])
print("\nMean of arr12:\n", arr12.mean())
print("\nSum of arr12:\n", arr12.sum())
print("\nStandard deviation of arr12:\n", arr12.std())

# Transposing arrays
print("\nTransposing arr8:\n", arr8.T)

# Dot product of two arrays
arr13 = np.array([1, 2])
arr14 = np.array([3, 4])
print("\nDot product of arr13 and arr14:\n", np.dot(arr13, arr14))

# Stacking arrays
arr15 = np.array([1, 2, 3])
arr16 = np.array([4, 5, 6])
print("\nVertical stacking of arr15 and arr16:\n", np.vstack((arr15, arr16)))
print("\nHorizontal stacking of arr15 and arr16:\n", np.hstack((arr15, arr16)))

# Splitting arrays
arr17 = np.arange(10)
print("\nSplitting arr17 into 2 parts:\n", np.array_split(arr17, 2))

# Broadcasting
arr18 = np.array([[1, 2, 3], [4, 5, 6]])
arr19 = np.array([1, 0, 1])
print("\nBroadcasting arr19 to arr18:\n", arr18 + arr19)
