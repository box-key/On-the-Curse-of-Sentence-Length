
import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t

def printN(int n):
  cdef int sum = 0
  for i in range(n):
    print(i)
    sum += i
  return sum

def getRange(int n):
  cdef np.ndarray arr = np.arange(n, dtype=DTYPE)
  cdef int sum = arr.sum()
  return arr, sum

def sumArr(np.ndarray[DTYPE_t, ndim=1] arr):
  cdef int length = len(arr)
  cdef int sum = 0
  for i in np.arange(length):
    sum += arr[i]
  return sum

def addHello(str s):
  return s + ' ' + 'Hello!'

def printLists(np.ndarray arr1, np.ndarray arr2):
  cdef int length = len(arr1)
  for i in np.arange(length):
    print(f'{arr1[i]} - {arr2[i]}')

def zeros(int n):
  cdef np.ndarray[DTYPE_t, ndim=1] temp = np.zeros((n,), dtype=DTYPE)
  return temp
