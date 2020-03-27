from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator cimport dereference as deref, preincrement as inc

cpdef vector[int] make_zeros(int num):
  cdef vector[int] zeros
  for i in range(num):
    zeros.push_back(0)
  return zeros

cpdef void assign_zeros(vector[int]& vec, int length):
  for i in range(length):
    vec[i] = 0

cpdef void test_assign_zeros(int num):
  cdef vector[int] test = range(num)
  print('Before')
  for t in test:
    print(t)
  assign_zeros(test, num)
  print('After')
  for t in test:
    print(t)


cpdef void print_num(int len):
  cdef vector[int] num
  for i in range(len):
    num.push_back(i)

  for i in range(len):
    print(num[i])

cpdef int print_name():
  uobj1 = u'Kei Nemoto'
  uobj2 = u'Kei Nemoto'
  cdef string name1 = <string> uobj1.encode('utf-8')
  cdef string name2 = <string> uobj2.encode('utf-8')
  cdef int result = name1.compare(name2)
  return result

cpdef int compare_str(str data1, str data2):
  cdef string name1 = <string> data1.encode('utf-8')
  cdef string name2 = <string> data2.encode('utf-8')
  cdef int result = name1.compare(name2)
  return result

cpdef void print_names(list names):
  cdef vector[string] _names
  _names.reserve(len(names))

  for name in names:
    _names.push_back(name.encode("utf-8"))

  for _name in _names:
    print(_name)

cpdef void check_names(list names, str target):
  cdef string _target = target.encode("utf-8")

  cdef vector[string] _names
  _names.reserve(len(names))

  for name in names:
    _names.push_back(name.encode("utf-8"))

  for idx, _name in enumerate(_names):
    if _name.compare(_target)==0:
      print(f'There you are @ {idx}')
