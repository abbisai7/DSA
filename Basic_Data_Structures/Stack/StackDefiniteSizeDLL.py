class Node:
    def __init__(self,data=None,prev=None,next=None):
        self.data = data
        self.prev = prev
        self.next = next

class DoubleLinkedList:
    
    #dummynode and cur 
    dummyNode = cur = Node(-1,None,None)
    curSize = 0
    def __init__(self,size) -> None:
        self.size = size

    def push(self,val):

        if self.curSize == self.size:
            self.pop()
        
        #create new node and set prev to cur
        node = Node(val,self.cur,None)
        #set cur's next to new node 
        self.cur.next = node
        #make cur as new node
        self.cur = node
        self.curSize +=1

        return

    def pop(self):
        #if no elements are present just return
        if self.dummyNode.next == None:
            return -1

        temp = self.cur.prev
        #cur's prev next to none
        self.cur.prev.next = None
        #set cur's prev to none
        self.cur.prev = None
        #make temp as cur
        self.cur = temp
        self.curSize -= 1

        return temp.data

    def peek(self):
        if self.dummyNode.next == None:
            return -1

        return self.cur.data

    def printDLL(self):

        if self.dummyNode.next == None:
            return -1

        temp = self.dummyNode.next
        s=''
        while temp!= None:
            s+= str(temp.data)+'->' if temp.next else str(temp.data)
            temp = temp.next
        return s


if __name__ == '__main__':
    stack = DoubleLinkedList(3)
    print(stack.printDLL())
    print(stack.pop())
    print(stack.push(1))
    print(stack.push(2))
    print(stack.push(3))
    print(stack.pop())
    print(stack.peek())
    print(stack.push(4))
    print(stack.printDLL())
    print(stack.push(5))
    print(stack.printDLL())
    print(stack.push(6))
    print(stack.printDLL())