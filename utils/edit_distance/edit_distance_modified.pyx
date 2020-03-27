from libcpp.vector cimport vector
from libcpp.string cimport string

cpdef vector[string] _copy_list(list data):
  cdef vector[string] _data
  for _str in data:
    _data.push_back(_str.encode("utf-8"))
  return _data

cpdef void assign_zeros(vector[int]& vec, int length):
  for i in range(length):
    vec[i] = 0

cpdef vector[int] make_zeros(int num):
  cdef vector[int] zeros
  for i in range(num):
    zeros.push_back(0)
  return zeros

cpdef int edit_distance_by_token(list sen1, list sen2):
  cdef int len_s1 = len(sen1)+1
  cdef int len_s2 = len(sen2)+1

  if len_s1==0 or len_s2==0:
    return max(len_s1, len_s2)

  if len_s1 > len_s2:
    sen1, sen2 = sen2, sen1
    len_s1, len_s2 = len_s2, len_s1

  cdef vector[string] _sen1 = _copy_list(sen1)
  cdef vector[string] _sen2 = _copy_list(sen2)

  cdef vector[int] distances = range(len_s1)
  distances.reserve(len_s1)

  cdef vector[int] dist_temp = make_zeros(len_s1)
  dist_temp.reserve(len_s1)

  cdef int comp
  cdef string w1,w2
  for i2, c2 in enumerate(_sen2):
    w2 = <string> c2
    dist_temp[0] = i2+1
    for i1, c1 in enumerate(_sen1):
      w1 = <string> c1
      comp = w1.compare(w2)
      dist_temp[i1+1] = distances[i1] if comp==0 else 1 + min((distances[i1], distances[i1+1], dist_temp[i1]))
    distances = dist_temp
    assign_zeros(dist_temp, len_s1)

  return distances.back()
