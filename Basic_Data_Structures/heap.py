import heapq

l = [5,3,1,9,7]

#heapify convert iterable to heap
#by default min heap is returned
#to use max heap push elements with negative sign
heapq.heapify(l)
print(l)

#heapush - pushes element to heap

heapq.heappush(l,2)
print(l)

#heappop - pop smallest from list
#basically 0th index element
print(heapq.heappop(l))
print(l)
print(heapq.heappop(l))

#heappushpop(heap, ele) - fist pushes into heap and then pop from the resultant
#if the element is already present nothing happens to heap
print(l)
print(heapq.heappushpop(l,2))#2
print(l)
#heapreplace(heap,ele)-first pop the element and then pushes to heap
print(l)#[3,7,5,9]
print(heapq.heapreplace(l,2))#3
print(l)#[2,7,5,9]

#nlargest(k, iterable, key = fun)
print(heapq.nlargest(3,l))

#nsmallest(k, iterable, key = fun)
print(heapq.nsmallest(4,l))

#return k  largest
l=[20,7,15,14,2,3]
heap = []
k=3
heapq.heapify(heap)
for i in l:
    heapq.heappush(heap,i)
    if len(heap)>k:
        heapq.heappop(heap)
print(heap)

