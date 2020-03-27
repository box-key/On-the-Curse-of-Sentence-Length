from libcpp.string cimport string
from libcpp.vector cimport vector

cdef print_name():
  cdef string name = b'Kei Nemoto'
  print('Hello', name)
