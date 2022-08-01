#Sorting

#Stability
#Stability in algorithms  means when we have repeated elements in input, if the elements appears in same order as in the input in the output.

#5(Red),7,5(yellow),2
#2,5(red),5(yellow),7

#Bubble Sort
#Bubble Sort Algorithm, works by repedately moving the largest element to last position.
#Bubbles the largest element to last place.

#Bubble sort is a stable sorting algorithm,and sorting takes places inplace

# A = [53,2,46,47,98,0,1]

# #original Algorithm
# def bubbleSort(A):
#     for i in range(len(A)):
#         for j in range(0,len(A)-1-i):
#             if A[j] > A[j+1]:
#                 A[j],A[j+1] = A[j+1],A[j]

# bubbleSort(A)
# print(A)

#improved Algorithm
#in the case of already sorted, we do all the checks in original algorithm
#so to break, we will introduce flag variable, if no swaps are done, we will set flag= False, which means array is sorted

#improved algorithm

# def improvedBubbleSort(A):
#     for i in range(len(A)):
#         flag = True
#         for j in range(0,len(A)-i-1):
#             if A[j]>A[j+1]:
#                 flag = False
#                 temp = A[j]
#                 A[j] = A[j+1]
#                 A[j+1] = temp

#         if flag:
#             return

# A=[5,1,2,3,4]
# improvedBubbleSort(A)
# print(A)

#BubbleSort Time Complexity Analaysis

#Worst Case and Average case of Complexity is O(N*N)
#Improved
#Best Case of Improved algorithm is O(N)
##Worst Case and Average case of Complexity of Improved Complexity is O(N*N)

#Space Complexity is O(1)

###############################################################################################

#Merge Sort

#Merge sort uses divide,conquer and combine paradigm

#divide - partitioing an n array to element in two n/2 size
#conquer - sorting two subarrays recursively using merge sort
#combine - merging two sorted subarrays of size n/2 into n size

#base case of mergesort if the len of array is 0 or 1, it is sorted


#Merge sort is a stable algorithm , and its sorting doesnt take inplace.

l = [1,6,8,2,7,4,5]

def merge(A,temp,beg,end,mid):
    #temp=[0]*(end+1)

    i = beg
    j = mid+1
    # we use beg to update temp values accordingly
    idx=beg
    while i<=mid and j<=end:
        if A[i] <= A[j]:
            temp[idx] = A[i]
            idx += 1
            i+=1
        else:
            temp[idx] = A[j]
            idx+=1
            j+=1
    #if i has completely traversered till mid,will check has if j anything left and will push into the temp
    if i>mid:
        while j<=end:
            temp[idx] = A[j]
            idx+=1
            j+=1
    #reverse of above
    else:
        while i<=mid:
            temp[idx] = A[i]
            idx+=1
            i+=1
    #either end+1 or idx
    for k in range(beg,end+1,1):
        A[k] = temp[k]
    

def mergeSort(A,temp,beg,end):
    #run til end is greater than beginning
    if beg<end:
        mid = (beg+end)//2
        mergeSort(A,temp,beg,mid)
        mergeSort(A,temp,mid+1,end)
        merge(A,temp,beg,end,mid)

temp =[0]*len(l)
mergeSort(l,temp,0,len(l)-1)
print(l)

#Time Complexity
#Merge sort takes time complexity of O(nlogn)
#Space complexity is O(n) as we use temp 


