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
