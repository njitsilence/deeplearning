# import numpy as np
#
# # li = [1,2,3]
# # array = np.arange(10,20,2)
# #
# # print(li)
# # print(array.ndim,array.shape,array.size)
# #
# # print(array)
# #
# # print(np.arange(10))
#
# a = np.array([10,20,30,40])
# b = np.arange(10).reshape(2,5)
# # print(a,'\n',b)
# """
# [10 20 30 40]
#  [0 1 2 3]
# """
#
# print(b)
# print(np.max(b,axis=1))
#
#
# a = [1,2,3,4]
# b = [2,2,2,2]
# aa = np.array(a)
# bb = np.array(b)
#
# print(np.hstack((aa,bb)))
#
# a = np.arange(1,61).reshape((3,5,4))
#
# print(a)

import numpy as np

# a = np.array([1,2,3,4])
#
# print(a)
#
# import time
# a = np.random.rand(1000000)
# b = np.random.rand(1000000)
#
# tic = time.time()
# c = np.dot(a,b)
# toc = time.time()
# print(c)
# print(toc-tic)
#
# c = 0
#
# tic = time.time()
# for i in range(1000000):
#     c += a[i]*b[i]
# toc = time.time()
# print(c)
# print(toc-tic)


# a = [1,2,3,4]
# print(a)
# b = np.array(a)
# print(b+1)
#
# a = [[1,1,1],[2,2,2]]
# b = [3,3,3]
#
# aa = np.array(a)
# bb = np.array(b)
#
# print(a+b)
# print(aa+bb)

# np.random.seed = 1
# a = np.random.randn(5,1)
# print(a)
#
# print(a.T)
#
#
# b = np.array([ 0.02583403, 1.44833825,  0.2989536,  -0.1298582,  -0.20203427])
#
# print(np.dot(a,a.T))



# import pandas as pd
#
# pd.read_excel('test.xls')

# a = np.random.randn(2,2,2,5,2)
# print(a)

# l1 = [1,2,3,4,5,6,7,8]
#
# print(l1[:5])
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])

print(w)
print(w.shape)
print(X)
print(X.shape)
print('======')
print(w*X)
print('======')
print(np.dot(w.T,))


