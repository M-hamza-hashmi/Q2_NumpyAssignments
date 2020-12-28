#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[13]:


arr1=np.empty(10)
arr1


# 3. Create a vector with values ranging from 10 to 49

# In[14]:


arr2=np.arange(10,49)
arr2


# 4. Find the shape of previous array in question 3

# In[15]:


np.shape(arr2)


# 5. Print the type of the previous array in question 3

# In[16]:


type(arr2)


# 6. Print the numpy version and the configuration
# 

# In[12]:


print(np.__version__, np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[20]:


arr2.ndim


# 8. Create a boolean array with all the True values

# In[28]:


a=np.array(range(1,5),dtype="bool")
a


# 9. Create a two dimensional array
# 
# 
# 

# In[29]:


a2d=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a2d)
a2d.ndim


# 10. Create a three dimensional array
# 
# 

# In[30]:


a2d=np.array([[[1,2,3],[4,5,6],[7,8,9]],[[4,2,3],[4,8,6],[7,7,9]]])
print(a2d)
a2d.ndim


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[56]:


arr4=np.arange(10)
print(arr4)
arr4rev=arr4[-1:-11:-1]
print(arr4rev)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[2]:


arr5=np.empty(10,dtype='int32')
arr5[4]=1
arr5


# 13. Create a 3x3 identity matrix

# In[3]:


ide=np.eye(3)
ide


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[12]:


arr = np.array([1, 2, 3, 4, 5],dtype='float32')
print(arr)
arr.dtype


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[15]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
arr_m=arr1*arr2 #element wise 
print(arr_m)


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[19]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
arr_c= (arr1<arr2)
arr_c


# 17. Extract all odd numbers from arr with values(0-9)

# In[30]:


arr=np.arange(10)
arr_odd=arr[arr%2==1]
arr_odd


# 18. Replace all odd numbers to -1 from previous array

# In[32]:


arr[arr%2==1]=-1
arr


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[33]:


arr=np.arange(10)
arr[5:9]=12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[34]:


arr2d=np.array([[1,1,1],[1,0,1],[1,1,1]])
print(arr2d)


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[36]:


arr2d = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
arr2d[1,1]=12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[41]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0,:,:]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[43]:


arr2d=np.arange(9).reshape(3,3)
arr_s=arr2d[0,0:]
arr_s


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[44]:


arr2d=np.arange(9).reshape(3,3)
arr_s=arr2d[1,1]
arr_s


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[45]:


arr2d=np.arange(9).reshape(3,3)
arr_s=arr2d[0:2,2]
arr_s


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[29]:


arr=np.random.randn(10,10)*10
arr_min=arr.min()
print(f'Minimum Value of array is = ',arr_min)
arr_max=arr.max()
print(f'Maximum Value of array is = ',arr_max)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[39]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
arr=np.intersect1d(a,b) #finidng common elements by intersect1d
arr


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[41]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a==b)   #find position that where condition meets =,>,<,etc any  


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[44]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
data_notWill_pos=data[names!='Will']
data_notWill_pos


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[52]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
pos=np.where((names!='Will') & (names!='Joe'))
pos
data_notjoe_will=data[pos]
data_notjoe_will


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[66]:


arr=np.random.uniform (low=1.0, high=15.0, size=(5,3))
arr


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[67]:


arr=np.random.uniform (low=1.0, high=16.0, size=(2,2,4))
arr


# 33. Swap axes of the array you created in Question 32

# In[68]:


arr_n=np.transpose(arr)
arr_n


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[81]:


arr=np.linspace(0,1,10)
arr_sqrt=np.sqrt(arr)
arr_sqrt[arr_sqrt<0.5]=0
arr_sqrt


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[8]:


import random
arr1=np.random.randint (low=0, high=12, size=(12))
print(arr1)
arr2=np.random.randint (low=0, high=12, size=(12))
print(arr2)
arr3=arr1[arr1>arr2]
print(arr3)
arr4=arr2[arr2>arr1]
print(arr4)
arr_c=np.append(arr3,arr4)
print(f'array with maximum values between each element of the two arrays', arr_c)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[9]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
unique_n=np.unique(names)
print(unique_n)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[19]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
n=np.intersect1d(a,b)
a = filter(lambda x: x if x not in n else None, a)
a=np.array(list(a))
print(a)
           


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[26]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
del_samplearray=np.delete(sampleArray,1,axis=1)
newColumn = np.array([[10,10,10]])
newColumn=newColumn.transpose()
new_samplearray = np.insert(del_samplearray, 1, values=newColumn[:,0], axis=1)
new_samplearray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[27]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
dp=np.dot(x,y)
dp


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[37]:


arr1=np.random.randint (low=1, high=15, size=(20))
print(arr1)
arr1.cumsum()

