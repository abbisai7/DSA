#Day-1

#1. Set Matrix Zeros

#https://leetcode.com/problems/set-matrix-zeroes/

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        #TC - O(2*(N+M))
        #SC - O(1)
        rows = len(matrix)
        cols = len(matrix[0])
        #to store whether any rows 0th cols values are zero or not
        col0 = False
        for row in range(rows):
            #if 0th column value is 0 set flag col0 to True
            if matrix[row][0]==0:
                col0 = True
            #we will check from 1 cols in every row, since 0 th columsn elements are overlapping
            #(2,0) =0 by below condition it will also make (0,2) as 0 which is false
            for col in range(1,cols):
                #if 0 set it 0th row and 0th col as 0
                if matrix[row][col]==0:
                    matrix[row][0] = 0
                    matrix[0][col] = 0
        
        #again traverse in reverse direction
        for row in range(rows-1,-1,-1):
            #for any row only till 1th column
            for col in range(cols-1,0,-1):
                #for any (row,col) pair if either of its 0th row or 0th col is 0 set its value to 0
                if matrix[row][0] == 0 or matrix[0][col] == 0:
                    matrix[row][col] = 0
            #for a row, if col are iterated till, check for col0 if true update it to 0        
            if col0 == True:
                matrix[row][0] = 0
                    
        
        #1. First Method
        #TC - O(M*N)
        #SC - O(M*N)
#         l = []
        
#         #push the (row,col) pair into list whose values are 0
#         for i in range(len(matrix)):
#             for j in range(len(matrix[0])):
#                 if matrix[i][j] == 0:
#                     l.append((i,j))
                    
        
#         while l:
#             x,y = l.pop()
            
#             #set col values to 0
#             for i in range(len(matrix)):
#                 matrix[i][y] = 0
             
#             #set row values to 0
#             for j in range(len(matrix[0])):
#                 matrix[x][j] = 0



#2. Pascal's Triangle

#https://leetcode.com/problems/pascals-triangle/

class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        
        #Striver's Method
        res =[]
        for i in range(numRows):
            row = []
            for j in range(i+1):
                if j==0 or j==i:
                    row.append(1)
                else:
                    row.append(res[i-1][j-1]+res[i-1][j])       
            res.append(row)
                    
        return res
    
        #TC-O(N*N)
        #SC-O(N*N)
        #intilaize list of list initially with 1
        res = [[1]*i for i in range(1,numRows+1)]
        for i in range(numRows):
            #0th index and 1th index values remains same
            #so it starts iterating for 2nd row till ith value
            for j in range(1,i):
                #prev row elements
                res[i][j] = res[i-1][j-1]+res[i-1][j]
                
        return res


#3. Next Permutation
# 
# https://leetcode.com/problems/next-permutation/
# 
# class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        
        #TC - O(N)
        #SC - O(1)
        
        if len(nums) == 0 or len(nums)<=1:
            return
        
        #initializing i with last before element in array
        #i is used to find break b/w increasing and decreasing sequence
        #in the case 1,2,3,4,5
        #we take 4 as break
        i = len(nums)-2
        
        #iterate till we find curElem less than its next element
        while i>=0 and nums[i]>=nums[i+1]:
            i -= 1
        
        #this check is required because in case of 3,2,1 i will be -1, in this case we will
        #just reverse list as outuput
        if i>=0:
            j = len(nums)-1
        
            #find greater element than ith index element
            while nums[j]<=nums[i]:
                j -= 1
        
            #swap elements
            nums[i], nums[j] = nums[j],nums[i]
        
        #then reverse the array from i+1th index to last
        i+=1
        n=len(nums)-1
        #n because middle element remains same in reverse
        while(i<n):
            nums[i],nums[n] = nums[n],nums[i]
            i+=1
            n-=1


#4.Maximum Subarray
# 
# https://leetcode.com/problems/maximum-subarray/
# 
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        #TC - O(N)
        #SC - O(1)
        
        #Kadanes Algorithm
        totalSum = 0
        #to handle negative values
        #for example in case -1 max will be -1,if we use it will updated to 0
        maxSum = -1000001
        
        for i in nums:
            
            totalSum += i
            maxSum = max(totalSum,maxSum)
            #if totalSum is less than 0, update it to 0
            if totalSum < 0:
                totalSum = 0
        
        return maxSum 


#5. Sort Colors
#https://leetcode.com/problems/sort-colors/

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        #Dutch National Flag algorithm
        #TC - O(N)
        #SC - O(1)
        #a[0,low-1]-0's
        #a[high+1,n]-2's
        #middle we will have 1's
        low = 0
        mid = 0
        high = len(nums)-1
        
        while mid<=high:
            
            #if mid is 0 swap both low and mid pointer values and increment their pos
            if nums[mid] == 0:
                nums[low],nums[mid] = nums[mid],nums[low]
                low+=1
                mid+=1
            
            #if mid is 1 just increment the index
            elif nums[mid] == 1:
                mid+=1
            
            #if mid is 2 swap it with high, and just decrement high
            elif nums[mid] == 2:
                nums[mid], nums[high] = nums[high],nums[mid]
                high-=1
                
                
                
        #TC - O(N)+O(N)
        #SC - O(1)
        count0 =0
        count1 = 0
        count2 = 0
        
        for i in nums:
            if i==0:
                count0 +=1
            elif i==1:
                count1 +=1
                
            else:
                count2 += 1
        i=0        
        while count0>0:
            nums[i]=0
            count0 -= 1
            i+=1
        
        while count1>0:
            nums[i]=1
            i+=1
            count1 -=1
            
        while count2>0:
            nums[i]=2
            i+=1
            count2 -=1



#6.Best Time to Buy and Sell Stock
#https://leetcode.com/problems/best-time-to-buy-and-sell-stock/


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        
        #Striiver approach
        #TC - O(N)
        #SC - O(1)
        minValue = 100001
        maxPrice = 0
        
        for i in prices:
            #store the value in the array
            minValue = min(minValue,i)
            #minus every element with minValue and store max of it
            maxPrice = max(maxPrice,i-minValue)
            
        return maxPrice
#         #Brute Foce
#         #TC - O(N*N)
#         #SC - O(1)
#         maxProfit = 0
        
#         for i in range(len(prices)):
#             for j in range(i,len(prices)):
#                 if prices[j]>prices[i]:
#                     maxProfit=max(maxProfit,prices[j]-prices[i])
                    
#         return maxProfit
        


#Day-2(13/07/2022)

#1.Rotate Image

#https://leetcode.com/problems/rotate-image/

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
#         #TC - O(N*N)+O(1)
#         #SC - O(1)
        
#         #first calculating transpose of the matrix
#         for i in range(len(matrix)):
#             #using i because instead of 0 because it may reswap elements
#             for j in range(i,len(matrix[0])):
#                 matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
                
#         #and then reverse each row to get ouptut     
#         for i in range(len(matrix)):
#             matrix[i].reverse()
                
                
        #Brute Force
        #TC - O(N*N)
        #SC - O(N*N)
        
        res = [[-1 for _ in range(len(matrix[0]))]for i in range(len(matrix))]
        n=len(matrix)-1
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                res[i][j] = matrix[n-j][i]
                print(res[i][j])


#2 Merge Intervals

#https://leetcode.com/problems/merge-intervals/

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        
        #Strivers approach
        #TC - O(N)+O(NLogN)
        
        res = []
        
        #sorting the list based on first elemnet
        intervals.sort(key = lambda x:x[0])
        
        #start first elem of first elem
        start = intervals[0][0]
        #second elem of firsrt elem
        end = intervals[0][1]
        
        #iterarting over the interavls
        for i in intervals:
            #if current List start is less than ennd then update end
            if i[0] <= end:
                end = max(end,i[1])
            #push the currents start and end to res and update start and end with current Pair
            else:
                res.append([start,end])
                start = i[0]
                end = i[1]
        #[[1,5]]- edge case
        res.append([start,end])
        
        return res
        
        #Brute Force
        #TC - O(NLogn)+O(N*N)
        #SC - O(N)
        res =[]
        #first we sort based on first element of list
        intervals.sort(key=lambda x:x[0])
        
        #then iterate over list
        for i in range(len(intervals)):
            
            start = intervals[i][0]
            end = intervals[i][1]
            
            #if already there is a value in res
            if res:
                #check whether current Interval has already been merged
                #current Start is less than last pushed elemet end
                if start <= res[-1][1]:
                    continue
                    
                    
            for j in range(i+1,len(intervals)):
                #if start is less than previous end
                if intervals[j][0]<=end:
                    #checking for max end
                    end = max(end,intervals[j][1])
            #at last push the pair        
            res.append([start,end])
            
        return res



#3.Merge Sorted Array

#https://leetcode.com/problems/merge-sorted-array/


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        #TC - O(M+N)
        #SC - O(1)
        
        #last positions
        j=n-1
        #given non 0 value positon
        i=m-1
        k=len(nums1)-1
        
        #same as merging in merge sort
        #but we will move from back side
        while i>=0 and j>=0:
            #if second row elements is greater than non 0th element push it to back and decrement j
            if nums2[j]>nums1[i]:
                nums1[k]=nums2[j]
                j-=1
            #if non 0th element is greater than nusm2 the element push non 0th element current pos
            else:
                nums1[k] = nums1[i]
                i-=1
            k-=1
        
        #anything unvisited push it to the front
        while i>=0:
            nums1[k]= nums1[i]
            i-=1
            k-=1
        
        while j>=0:
            nums1[k] = nums2[j]
            j-=1
            k-=1
                
                
        
            
            
        
#         #TC - O(m+n)log(m+n)
#         #SC - O(1)
#         j=0
#         for i in range(m,m+n):
#             nums1[i] = nums2[j]
#             j+=1
            
#         nums1.sort()



#Merge sorted Array another version

#https://practice.geeksforgeeks.org/problems/merge-two-sorted-arrays-1587115620/1#

import math
class Solution:
    
    #Function to merge the arrays.
    def merge(self,arr1,arr2,n,m):
        #code here
        
        #Gap Method
        
        #TC - O(nlogn)
        
        t = n+m
        #we will take ceil 
        gap = math.ceil(t/2)
        #iterate until gap is 0
        while gap>0:
            i=0
            j=gap
            
            while j<n+m:
                if j<n and arr1[i]>arr1[j]:
                    arr1[i], arr1[j] = arr1[j], arr1[i]
                #j-n beacuse arr2 has index 0 so accesing the elemnet based on first arry lenght
                elif i<n and j>=n and arr1[i]>arr2[j-n]:
                    arr1[i],arr2[j-n] = arr2[j-n],arr1[i]
                elif i>=n and j>=n and arr2[i-n]>arr2[j-n]:
                    arr2[i-n], arr2[j-n] = arr2[j-n],arr2[i-n]
                i+=1
                j+=1
            #for 1 we will assign 0 directly as ceil(1/2) gives 2
            if gap == 1:
                gap = 0
            else:
                gap = math.ceil(gap/2)
            
        
        #TC - O(n*m)
        #SC - O(1)
        # for i in range(n):
        #     #if current element of first array is greater than first element of
        #     if arr1[i]>arr2[0]:
        #         arr1[i],arr2[0] = arr2[0], arr1[i]
            
        #     #pushing the swapped element to correct position in arr2
        #     first = arr2[0]
        #     k=1
        #     while k<m and arr2[k]<first:
        #         arr2[k-1] = arr2[k]
        #         k+=1
                    
        #     arr2[k-1] = first
        
        
                    
            
        #Brute Force
        # #TC - O((m+n)log(m+n))
        # #SC - O(M+N)
        # arr3=[]
        
        # for i in arr1:
        #     arr3.append(i)
            
        # for i in arr2:
        #     arr3.append(i)
            
        
        # arr3.sort()
        # k=0
        # for i in range(len(arr1)):
        #     arr1[i] = arr3[k]
        #     k+=1
        
        
        # for i in range(len(arr2)):
        #     arr2[i] = arr3[k]
        #     k+=1
            
        # return arr3

#4.Find the Duplicate Number

#https://leetcode.com/problems/find-the-duplicate-number/


class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        
        #TC - O(N)
        #SC - O(1)
    
        #intilaize this first index value
        slow = nums[0]
        fast = nums[0]
        
        #tortise method
        while True:
            #slow will move one value at a time
            slow = nums[slow]
            #fast will move two value at a time
            fast = nums[nums[fast]]
            #if both collide
            if slow == fast:
                break
        #after collision assign fast to start        
        fast = nums[0]
        #traverse by moving slow and fast at one value at a time
        while slow!=fast:
            slow = nums[slow]
            fast = nums[fast]
            
        return slow
            
        
        #Brute Force
        #TC - O(NLogN)+O(N)
        #SC - O(1)
        nums.sort()
        
        for i in range(len(nums)-1):
            if nums[i] == nums[i+1]:
                return nums[i]

#5.Repeat and Missing Number

#https://www.codingninjas.com/codestudio/problems/615?topList=striver-sde-sheet-problems&utm_source=striver&utm_medium=website


#the inttion here is to calculate sum of numbers and both square of numbers

#sum of numbers(x)
#sum of given numbers array(y)
#x-y gives difference b/w missing and repeatig number
#x^2-y^2 == (x+y)(x-y) == s2
#x-y = s1
#(x+y)(x-y) = s2
#x+y = s2/s1

#(x+y)+(x-y) => x= (s1+s2/s1)//2
#where x is repeating number


def missingAndRepeating(A, n):
    # Write your code here
    s1 = (n*(n+1))//2
    s2 = (n*(n+1)*((2*n)+1))//6
    
    #minus number and square from total sum and sum of squares
    for i in A:
        s1 -= i
        s2 -= i*i

    #can be get by above equation    
    miss = (s1+(s2//s1))//2
    
    repeat = miss-s1
    
    return [miss,repeat]

#6.Count Inversions

#https://practice.geeksforgeeks.org/problems/inversion-of-array-1587115620/1Count%20Inversions

class Solution:
    #User function Template for python3
    
    # arr[]: Input Array
    # N : Size of the Array arr[]
    #Function to count inversions in the array.
    def inversionCount(self, arr, n):
        # Your Code Here
        #TC - O(NLogN)
        #SC - O(N)
        temp =[0]*n
        return self.mergeSort(arr,temp,0,n-1)
        
    def mergeSort(self,arr,temp,start,end):
        inverseCount = 0
        if start<end:
            mid = (start+end)//2
            inverseCount += self.mergeSort(arr,temp,start,mid)
            inverseCount += self.mergeSort(arr,temp,mid+1,end)
            inverseCount += self.merge(arr,temp,start,end,mid)
        return inverseCount
        
    
    def merge(self,arr,temp,start,end,mid):
        inverseCount = 0
        
        i = start
        j = mid+1
        k = start
        while i<=mid and j<=end:
            if arr[i]<=arr[j]:
                temp[k] = arr[i]
                i+=1
            else:
                temp[k] = arr[j]
                j+=1
                #only change compared to merge sort code
                #for example two list [5,6,7] [2,3]
                #for 5<2 implies 2 will form pair with all elements in left,which is given mid-i+1
                #+1 because it 0 based index
                inverseCount += (mid-i+1)
            k+=1
        if i>mid:
            while j<=end:
                temp[k] = arr[j]
                j+=1
                k+=1
   
        else:
            while i<=mid:
                temp[k] = arr[i]
                i+=1
                k+=1
        
     
        for z in range(start,end+1):
            arr[z] = temp[z]
        
        return inverseCount




