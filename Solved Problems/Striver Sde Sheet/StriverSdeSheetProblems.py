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


###################################################################

#Day-3(July 21 2022)


#1.Search a 2D Matrix

#https://leetcode.com/problems/search-a-2d-matrix/

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        
        #Binary Search
        #consider given matrix as an array peform binary search 
        m = len(matrix)
        n = len(matrix[0])
        
        start = 0
        end = (m*n)-1
            
        while start<=end:
            
            mid = (start+end)//2
            
            #division by col size and modulo give mid's row and col
            row = mid//n
            col = mid%n
            
            if matrix[row][col] == target:
                return True
            
            if target<matrix[row][col]:
                end = mid-1
            else:
                start = mid+1
                
        return False
            
        #TC - O(m+n)
        #SC - O(1)
#         m = len(matrix)
#         n = len(matrix[0])
        
#         start = 0
#         end = n-1
        
        
#         while start<m and end>=0:
            
#             if matrix[start][end] == target:
#                 return True
            
#             if matrix[start][end]>target:
#                 end -= 1
            
#             else:
#                 start += 1
                
#         return False
        

#2.Pow(x, n)

#https://leetcode.com/problems/powx-n/


class Solution:
    def myPow(self, x: float, n: int) -> float:
        
        #striver method
        
        #TC - O(Logn)
        #SC - O(1)
        
        #we take absolute value of n
        nn = abs(n)
        res = 1
        while nn>0:
            #if power is divisible we X^n = (x^2)n/2
            #which makes multiply x*x and dividin power by2
            if nn%2 == 0:
                x = x*x
                nn = nn//2
            #in the case of odd power n^5 = n*n^4
            else:
                res *= x
                nn = nn-1
                
        if n<0:
            return 1/res
        return res
        
        
        
        
#         #BRUTE FORCE
#         #TC -O(N)
#         #SC - O(1)
#         res = 1
#         for i in range(1,abs(n)+1):
#             res *= x
#         if n<0:
#             return 1/res
        
#         return res
        

#3.Majority Element


#https://leetcode.com/problems/majority-element/


class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        
        #Morees's voting algo
        #TC - O(N)
        #SC - O(1)
        #In this problem sure there will be only one majority element
        ele = 0
        count = 0
        
        for i in nums:
            
            #if the count is 0 set ele to current elem
            if count == 0:
                ele = i
            #curEle and elem are same increase count by 1 else decrease by 1    
            if i == ele:
                count += 1
            else:
                count -= 1
        
        return ele
        
        
        
#         #TC - O(N)
#         #SC - O(N)
#         d={}
        
#         for i in nums:
#             if i in d:
#                 d[i] +=1
#             else:
#                 d[i] = 1
                
#         ans=0
        
#         for i in d.keys():
            
#             if d[i]>= len(nums)/2:
#                 ans=i
                
#         return ans
        

#4.Majority Element II

#https://leetcode.com/problems/majority-element-ii/


class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        
        #TC - O(N)
        #SC - O(1)
        #Since we need to check above n/3 there can be only two majority elements
        count1=0
        ele1=-1
        count2=0
        ele2=-1
        
        for i in nums:
            
            if i == ele1:
                count1 += 1
                
            elif i == ele2:
                count2 += 1
            
            elif count1 == 0:
                ele1 = i
                count1 = 1
            
            elif count2 == 0:
                ele2 = i
                count2 = 1
                
            else:
                count1 -= 1
                count2 -= 1
        
        
        ans = []
        num1=0
        num2=0
        for i in nums:
            if ele1 == i:
                num1 +=1
            elif ele2 == i:
                num2 +=1
        
        n = len(nums)
        if num1 > (n/3):
            ans.append(ele1)
        if num2 > (n/3):
            ans.append(ele2)
            
        return ans
            
            
            
            
            
        
        
#         #TC -O(N)
#         #SC -O(N)
#         d={}
        
#         for i in nums:
#             if i in d:
#                 d[i] +=1
#             else:
#                 d[i] = 1
                
#         ans=[]
        
#         for i in d.keys():
            
#             if d[i]>len(nums)/3:
#                 ans.append(i)
                
#         return ans
        


#5.Unique Paths

#https://leetcode.com/problems/unique-paths/

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        
        #Strivers Combination Approach
        
        #TC - O(m-1) if we row count or O(N-1) if we col count
        #SC - O(1)
        res = 1
        stepCount = m+n-2
        #take row count
        row = m-1
        
        #this for loop count ncr 
        #10c3 can be 10*9*8/1*2*3
        for i in range(1,row+1):
            res *= ((stepCount-row)+i)/i
            
        return round(res)
        
        #Dynamic Programming Approach
        #TC - O(m*n) 
        #SC - O(m*n)
#         return self.countPaths(0,0,m,n,{})

#     def countPaths(self,curRow,curCol,m,n,memo):
        
#         if curRow == m-1 and curCol == n-1:
#             return 1
        
#         if curRow >=m or curCol>=n:
#             return 0
        
#         curKey = str(curRow) + '-' + str(curCol)
        
#         if curKey in memo:
#             return memo[curKey]
        
        
#         right = self.countPaths(curRow,curCol+1,m,n,memo)
#         down = self.countPaths(curRow+1,curCol,m,n,memo)
        
#         memo[curKey] = right+down
        
#         return memo[curKey]












#6.Reverse Pairs


#https://leetcode.com/problems/reverse-pairs/

class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        #TC - O(NLogn)+O(N)+O(N)
        #SC - O(N)
        temp = [0]*len(nums)
        return self.mergeSort(nums,temp,0,len(nums)-1)
    
    
    def mergeSort(self,nums,temp,start,end):
        
        inv = 0
        if start<end:
            mid = (start+end)//2
            
            inv += self.mergeSort(nums,temp,start,mid)
            inv+=self.mergeSort(nums,temp,mid+1,end)
            inv+=self.merge(nums,temp,start,mid,end)
            
        return inv
            
    def merge(self,nums,temp,start,mid,end):
            
        inv = 0
        #logic apart from merge sort
        y = mid+1
        #iterate first part of array over second
        for x in range(start,mid+1):
            #if part array ele is greater than 2*nums of second part
            while y<=end and nums[x]>2*nums[y]:
                y+=1
            
            #curret y - mid+1 since mid is 0 index
            inv += y-(mid+1)
            
            
        i = start
        j = mid+1
        k = start
        
        while i<= mid and j<=end:
            if nums[i]<=nums[j]:
                temp[k] = nums[i]
                i += 1
            else:
                temp[k] = nums[j]
                j += 1
                
            k += 1
        
        if i>mid:
            while j<=end:
                temp[k] = nums[j]
                j += 1
                k += 1
        else:
            while i<=mid:
                temp[k] = nums[i]
                i += 1
                k += 1
                
        for u in range(start,end+1):
            nums[u] = temp[u]
            
        return inv
        
        
        
        
        #Brute Force
        #TC - O(N*N
        #SC - O(1)
#         count = 0
#         for i in range(len(nums)):
#             for j in range(i+1,len(nums)):
                
#                 if nums[i]>2*nums[j]:
#                     count+=1
                    
#         return count
        


#Day-4


#1.Two Sum

#https://leetcode.com/problems/two-sum/


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        #TC - O(N)
        #SC - O(N)
        
        d = {}
        
        for i in range(len(nums)):
            comp = target-nums[i]
            
            if comp in d:
                return [d[comp],i]
            
            else:
                d[nums[i]] = i

#2. 4 Sum

#https://leetcode.com/problems/4sum/

class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        
        #TC - O(NLogn)+O(N^3)
        #SC - O(1)
        res = []
        n = len(nums)
        
        #sort the numbers
        nums.sort()
        i=0
        while i<len(nums):
            target1 = target-nums[i]
            j=i+1
            while j<len(nums):
                target2 = target1-nums[j]
                
                front = j+1
                back = n-1
                
                while front < back:
                    
                    two_sum = nums[front]+nums[back]
                    
                    #4<7(target)
                    #move front else back
                    if two_sum<target2:
                        front += 1
                        
                    elif two_sum>target2:
                        back -= 1
                        
                    else:
                        
                        quad = []
                        
                        quad.append(nums[i])
                        quad.append(nums[j])
                        quad.append(nums[front])
                        quad.append(nums[back])
                        res.append(quad)
                        

                        
                        #to overcome the duplicate
                        #current at 3,3,3,4 pushes to four
                        while front<back and nums[front]==quad[2]:
                            front+=1
                        
                        #current at 4 3,3,3,4,4,4 pushes back to 3 
                        while front<back and nums[back]==quad[3]:
    
                            back -= 1
                
                #same iterate till till last element of duplcuate
                #2,2,2,3 goes till last 2
                while j+1<n and nums[j]==nums[j+1]:
                    j+=1
                    
                j+=1
            
                    
            
            
            while i+1<n and nums[i] == nums[i+1]:
                i+=1
                
            i+=1
                
                
        return res
                
                

#3. Longest Consecutive Sequence


#https://leetcode.com/problems/longest-consecutive-sequence/

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        
        #striver method
        
        #TC - O(N)
        #SC - O(N)
        
        #remove duplicates
        s = set(nums)
        
        res = 0
        for i in range(len(nums)):
            if nums[i]-1 in s:
                continue
            
            #starting of the sequence
            cur = nums[i]
            ans = 0
            while cur in s:
                ans+=1
                s.remove(cur)
                cur = cur+1
                
            res = max(ans,res)
            
        return res


#4.Largest subarray with 0 sum
# 
# https://practice.geeksforgeeks.org/problems/largest-subarray-with-0-sum/1
# 
# 
class Solution:
    def maxLen(self, n, arr):
        #Code here
        
        #TC - O(N)
        #SC - O(N)
        
        d = {}
        #to handle edge case and if total array has zero sum
        d[0] = -1
        
        ps = 0
        ans = 0
        
        for i in range(len(arr)):
            ps += arr[i]
            
            #if ps is in hashmap calculate distance curr i value and value in hashmap
            if ps in d:
                ans = max(ans,i-d[ps])
                
            else:
                d[ps] = i
                
        return ans
        
        #TC - O(N*N)
        #SC - O(1)
        # res = 0
        # for i in range(len(arr)):
        #     s=0
        #     for j in range(i,len(arr)):
        #         s+=arr[j]
                
        #         if s == 0:
        #             res = max(res,j-i+1)
                    
            
                    
        # return res      
        
        
        
        #Brute Force
        #TC - O(NLogN)+O(N)
        #SC - O(1)
#         if len(nums)==0:
#             return 0
        
#         nums.sort()
        
#         cur = 1
#         ans = 1
#         prev = nums[0]
        
#         for i in range(1,len(nums)):
            
#             if nums[i] == prev+1:
#                 cur += 1
            
#             #also take care of duplicates in input
#             #only in case of not equal set cur to 1
            
#             elif nums[i] != prev:
#                 cur = 1
                
#             prev = nums[i]
#             ans = max(ans,cur)
            
#         return ans



#5.Subarray with given XOR

#https://www.interviewbit.com/problems/subarray-with-given-xor/


class Solution:
    # @param A : list of integers
    # @param B : integer
    # @return an integer
    def solve(self, A, B):
        #TC - O(N)
        #SC - O(N)
        #same as leetcode 560
        #here we take xor instead of adding
        count = 0
        ps = 0
        #hashmap stores value and its freq
        d={}
        

        d[0] = 1
    
        for i in A:
            ps = ps^i
            
            #if xor of ps if already present return it freq        
            if ps^B in d:
                count += d[ps^B]
            if ps in d:
                d[ps] += 1
            else:
                d[ps] = 1
        return count


#6.Longest Substring Without Repeating Characters


#https://leetcode.com/problems/longest-substring-without-repeating-characters/

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        
        #Sahils Approach
        #TC - O(N)
        #SC - O(N)
        
        d = {}
        ans = 0
        release = 0
        
        for acquire in range(0,len(s)):
            
            #we can either of the below
            #running while curElement gets deleted from hashmap
            #abcc
            #firstrelease will be at 0 by the end of the loop it will be at 3
            # while release < acquire and d.get(s[acquire])!=None:
            #     d.pop(s[release])
            #     release +=1
            
            #if it is already present in hashmap and relase pointer is less than curElementIndex, we will change release curIndex+1, so that all element in new Window will be uniqu
            if d.get(s[acquire])!= None and release <= d[s[acquire]]:
                release = d[s[acquire]]+1                
                
            d[s[acquire]] = acquire
            ans = max(ans,acquire-release+1)
            
        return ans
        
        
        #Brute Force
        #TC - O(N*N)
        #SC - O(1)
        
#         if len(s) == 0:
#             return 0
#         ans = 1
        
#         for i in range(len(s)):
#             for j in range(i+1,len(s)+1):
#                 if len(s[i:j]) == len(set(s[i:j])):
#                     ans = max(ans,len(s[i:j]))
                
                
#         return ans


#Day-5
 # 
 # 
 # 1.Reverse Linked List


#https://leetcode.com/problems/reverse-linked-list/


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        #TC - O(N)
        #SC - O(1)
        prev = None
        
        while head:
            temp = head.next
            head.next = prev
            prev = head
            head = temp
            
        return prev



#2.Middle of the Linked List


#https://leetcode.com/problems/middle-of-the-linked-list/


class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        #TC - O(N)
        #SC - O(1)
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
        return slow



#3.Merge Two Sorted Lists


#https://leetcode.com/problems/merge-two-sorted-lists/

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        #TC - O(M+N)
        #SC - O(1)
        
        dummyPointer = currentPointer = ListNode(-1)
        
        while list1 and list2:
            if list1.val < list2.val:
                currentPointer.next = list1
                list1 = list1.next
            else:
                currentPointer.next = list2
                list2 = list2.next
            currentPointer = currentPointer.next
            
            
        while list1:
            currentPointer.next = list1
            list1 = list1.next
            currentPointer = currentPointer.next
            
        while list2:
            currentPointer.next = list2
            list2 = list2.next
            currentPointer = currentPointer.next
            
            
        return dummyPointer.next

#4.Remove N-th node from the end of a Linked List

#https://leetcode.com/problems/remove-nth-node-from-end-of-list/

class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        
        #TC - O(N)
        #SC - O(1)
        runner = head
        
        #dummy node to handle edge case in case when head itself to be deleted we need prev to map
        follower = dummyNode = ListNode(-1,head)
        
        #moving from 0 to n-1 first to place runner
        for i in range(n-1):
            runner = runner.next
        
        #runner will move till last n node, and follower will be n steps backwards,,which is at the point we need to delete
        while runner.next!=None:
            runner = runner.next
            follower = follower.next
        
        #mapping follower next's to next of next
        follower.next = follower.next.next
        
        #dummypointer hold head
        return dummyNode.next


#5.Add Two Numbers


#https://leetcode.com/problems/add-two-numbers/


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        
        #for example in sum 18
        #carry will be 1 which we get by 18//10
        #sum place will be 18%10 = 8
        
        #TC - O(N+M)
        #SC - O(1)
        
        dummyPtr = res = ListNode(-1)
        
        carry = 0
        
        while l1 and l2:
            s = l1.val+l2.val+carry
            carry = s//10
            
            node = ListNode(s%10)
            
            res.next = node
            
            l1 = l1.next
            l2 = l2.next
            res = res.next
            
        #if anything left
        while l1 != None:
            s=l1.val+carry
            carry = s//10
            node = ListNode(s%10)
            
            res.next = node
            l1=l1.next
            res = res.next
            
        while l2 != None:
            s=l2.val+carry
            carry = s//10
            node = ListNode(s%10)
            
            res.next = node
            l2=l2.next
            res = res.next
        
        #atlast if carry is not 0
        if carry==1:
            node = ListNode(carry)
            res.next = node
            
            
            
        return dummyPtr.next
            




#6.Delete Node in a Linked List

#https://leetcode.com/problems/delete-node-in-a-linked-list/


class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        #TC - O(1)
        #SC - O(1)
        
        node.val = node.next.val
        
        node.next = node.next.next



#Day 6(27th July 2022)


#1.Intersection of Two Linked Lists

#https://leetcode.com/problems/intersection-of-two-linked-lists/


class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        
        #Striver Method
        #TC - O(2*Max(M,N))
        #SC - O(1)
        
        d1 = headA
        d2 = headB
        
        while d1 != d2:
            d1 = headB if d1 == None else d1.next
            d2 = headA if d2 == None else d2.next
            
        return d1
        
        #Second Method
        #TC - O(M+N)
        #SC - (N)
        
#         s = set()
        
#         temp=headA
#         while temp!=None:
#             s.add(temp)
#             temp = temp.next
            
#         temp = headB
#         while temp != None:
#             if temp in s:
#                 return temp
#             temp = temp.next
            
#         return None
        
        #Brute Force
        #TC - O(M*N)
        #SC - O(1)
#         while headB!=None:
#             temp = headA
            
#             while temp!=None:
#                 if temp == headB:
#                     return temp
                
#                 temp = temp.next
                
#             headB = headB.next
            
#         return None




#2.Linked List Cycle

#https://leetcode.com/problems/linked-list-cycle/


class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        
        #TC - O(N)
        #SC - O(1)
        
        if head == None or head.next == None:
            return False
        
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                return True
            
        return False


#3. Reverse Nodes in k-Group


#https://leetcode.com/problems/reverse-nodes-in-k-group/


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        
        #TC - O(N/K)*k = O(N)
        #SC - O(1)
        if head == None and head.next == None:
            return None
        
        #first calculate the height
        height = self.getHeight(head)
        
        #have and dummy and prev
        dummyNode = ListNode(-1,head)
        prev = dummyNode
        
        #gives how many group to be done
        c = height//k
        
        
        while c>0:
            cur = prev.next
            nxt = cur.next
            #run till k-1
            #k=3-1=2
            #inp -> -1-1-2-3
            #step1 -> -1-2-1-3
            #step2 -> -1-3-2-1
            for i in range(k-1):
                
                cur.next = nxt.next
                nxt.next = prev.next
                prev.next = nxt
                nxt = cur.next
            c-=1
            prev = cur
            # cur = prev.next
            # nxt = cur.next
            
            
        return dummyNode.next
        
        
    def getHeight(self,head):
        count = 0
        
        while head:
            count+=1
            head = head.next
            
        return count
        

#4.Palindrome Linked List

#https://leetcode.com/problems/palindrome-linked-list/


class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        
        #TC - O(N/2)+O(N/2)+O(N/2)
        #SC - O(1)
        mid = self.getMid(head)
        
        reversedHalf = self.reverseHalf(mid)
        
        while head and reversedHalf:
            if head.val != reversedHalf.val:
                return False
            
            head = head.next
            reversedHalf = reversedHalf.next
            
        return True
    
    
    def getMid(self,head):
        
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
        return slow
    
    def reverseHalf(self,head):
        prev = None
        cur = head
        while cur:
            temp = cur.next
            cur.next = prev
            prev = cur
            cur = temp
            
        return prev



#5.  Linked List Cycle II

#https://leetcode.com/problems/linked-list-cycle-ii/


class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        #TC - O(N)
        #SC - O(1)
        if not head or not head.next:
            return None
        
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                fast = head
                while slow!=fast:
                    slow = slow.next
                    fast = fast.next
            
                return slow
        
        return None


#6. Flattening a Linked List

#https://practice.geeksforgeeks.org/problems/flattening-a-linked-list/1


def merge(n1,n2):
    
    dummyNode = temp = Node(-1)
    
    while n1!=None and n2!=None:
        
        if n1.data <= n2.data:
            temp.bottom = n1
            n1 = n1.bottom
        else:
            temp.bottom = n2
            n2 = n2.bottom
        
        temp = temp.bottom
        
    if n1 != None:
        temp.bottom = n1
    else:
        temp.bottom = n2
        
    return dummyNode.bottom

def flatten(root):
    #Your code here
    #TC - O(N)
    #SC - O(1)
    #we move to the last list and then start combining from last
    #for this recursion is used
    
    
    if root == None or root.next == None:
        return root
        
    
    #just normal merge in mergesort
    #we will keep on combining from end and pass the combined one to merge further
    root = merge(root,flatten(root.next))
    
    
    return root





#Day -7


#1.Rotate List

#https://leetcode.com/problems/rotate-list/

class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        
        #Striver Method
        
        #TC - O(N)+O(N-(k%N))
        #SC - O(1)
        if head == None or head.next == None:
            return head
        
        #count as 1 because we will come out before last node itself
        count = 1
        cur = head
        
        while cur.next != None:
            count += 1
            cur = cur.next
        
        cur.next = head
        
        
        #k%count remainder since k may be multiple of total length
        j = count - (k%count)
        
        #iterate last before k
        temp = head
        while j!=1:
            temp = temp.next
            j-=1
        #assign next of cur as head and cur next to null    
        head = temp.next
        temp.next = None
        
        return head
        #Brute Force
        #TC - O(N)+O(N*(k%N))
        #SC - O(1)
        
#         if head == None or head.next == None:
#             return head
        
#         count = 0
#         cur = head
#         while cur:
#             count += 1
#             cur = cur.next
            
#         #multiple of length of linked list give same as input
#         #so we just run modulo of k by length
#         for i in range(k%count):
#             cur = head
#             prev=None
#             while cur.next!=None:
#                 prev = cur
#                 cur = cur.next
#             prev.next = None
#             cur.next = head
#             head = cur
            
#         return head


#2.Copy List with Random Pointer

#https://leetcode.com/problems/copy-list-with-random-pointer/


class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        
        #Strivers Method
        #TC - O(N)
        #SC - O(1)
        cur = head
        #First round: make copy of each node,
        #and link them together side-by-side in a single list.
        #ex 1->2->3->None maps to 1->-1>2->-2->3->-3->None
        while cur:
            node = ListNode(cur.val)
            temp = cur.next
            cur.next = node
            node.next = temp
            cur = temp
            
        cur = head
        
        #now assign random pointer to new node created
        while cur:
            if cur.random:
                cur.next.random = cur.random.next
            else:
                cur.next.random = None
            cur = cur.next.next
            
        
        dummy = res =Node(-1)
        
        #now just link new nodes by excluding original nodes
        cur = head
        while cur:
            res.next = cur.next
            res = res.next
            cur = cur.next.next
            
        return dummy.next
        
        
        
        
        #Brute Force
        #TC - O(N)
        #SC - O(N)
        
        
        d={}
        
        cur = head
        d[None] = None
        #First create a node and store existing node as key and new node as value
        while cur:
            node = Node(cur.val)
            d[cur] = node
            cur = cur.next
        
        cur = head
        #for new node assign next and random as of original
        while cur:
            copy = d[cur]
            copy.next = d[cur.next]
            copy.random = d[cur.random]
            cur = cur.next
        
        #original head also will aslo be head in copy
        return d[head]

#3.3 Sum

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        
        #TC - O(N*N)
        #SC - O(M)- M no of triplets
        
        nums.sort()
        res = []
        
        for i in range(len(nums)-2):
            
            #to escape duplicate elements
            if i == 0 or (i>0 and nums[i]!=nums[i-1]):
                target = 0-(nums[i])
                
                low = i+1
                end = len(nums)-1
                
                while low<end:
                    
                    temp = nums[low]+nums[end]
                    if temp<target:
                        low+=1
                    elif temp>target:
                        end-=1
                    elif temp == target: 
                        res.append([nums[i],nums[low],nums[end]])
                        
                        #to escape duplicates
                        # while low<len(nums)-1 and nums[low]==nums[low+1]:
                        #     low+=1
                        while low<end and nums[low] == nums[low+1]:
                            low += 1
                        # while end>1 and nums[end]==nums[end-1]:
                        #     end-=1
                        
                        while low<end and nums[end]==nums[end-1]:
                            end-=1
                            
                        low+=1
                        end-=1
        
                        
        return res
        
        #Brute Force
        #TC - O(n*2logm)
        #SC - O(N)+O(M)
#         d = {}
#         res=[]
#         for i in nums:
#             if i in d:
#                 d[i]+=1
#             else:
#                 d[i] = 1
                
#         for i in range(len(nums)-2):
#             #not to include
#             d[nums[i]]-=1
#             for j in range(i+1,len(nums)-1):
#                 #not to include
#                 d[nums[j]]-=1
#                 temp = -(nums[i]+nums[j])
                
#                 if d.get(temp,0)>0:
#                     ans = sorted([nums[i],nums[j],temp])
#                     if ans not in res:
#                         res.append(ans)
#                 d[nums[j]]+=1
#             d[nums[i]]+=1
                    
        #return res
                    
        
        
        #Brute Force
        #TC - O(N^3logm)
        #SC - O(3*k)
#         res=[]
        
#         for i in range(len(nums)-2):
#             for j in range(i+1,len(nums)-1):
#                 for k in range(j+1,len(nums)):
#                     if nums[i]+nums[j]+nums[k] == 0:
#                         temp = sorted([nums[i],nums[j],nums[k]])
#                         res.append(temp)
                        
#         return res


#4. Trapping Rain Water

#https://leetcode.com/problems/trapping-rain-water/


class Solution:
    def trap(self, height: List[int]) -> int:
        #TC - O(N)
        #SC - O(1)
        #here in the middle of max trap, we should consider only space left by that index element, so at a instance maxlength - curheight should be added to res
        res = 0
        left = 0
        right = len(height)-1
        maxLeft = height[left]
        maxRight = height[right]
        while left<right:            
            
            if height[left]<height[right]:
                maxLeft = max(maxLeft,height[left])
                #res will have maxLeft - curheight as curHeight should nt be included in ans
                res+= maxLeft-height[left]
                left+=1
            else:
                maxRight = max(maxRight,height[right])
                res+= maxRight-height[right]
                right-=1
                
        return res


#5.Remove Duplicates from Sorted Array

#https://leetcode.com/problems/remove-duplicates-from-sorted-array/

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        #TC - O(N)
        #SC - O(1)
        
        #take 1 as we leave frist leave untouched and we place next elemet alos index
        i = 1
        for j in range(len(nums)-1):
            
            if nums[j] != nums[j+1]:
                #inplace replace
                nums[i] = nums[j+1]
                i+=1
        
        return i
        



#6.Max Consecutive Ones

#https://leetcode.com/problems/max-consecutive-ones/

class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        
        #second method
        
        #Tc - O(N)
        #SC - O(1)
        res = 0
        ans = 0
        
        for i in nums:
            #if it is one increment else mark res to 0
            if i == 1:
                res+=1
            else:
                res = 0
                
            ans = max(ans,res)
            
        return ans
        
#         #one method
#         #we will calculate the ps if previovs ps and currentPs is same
#         #we will max res in ans and mark res = 0
#         #if ps is not equal we will inc res
#         ps = 0
#         ans = 0
#         res = 0
#         for i in nums:
            
#             if ps+i == ps:
#                 ans = max(ans,res)
#                 res=0
#             else:
#                 res+=1
#             ps+=i
            
#         return max(ans,res)
    
    
    

