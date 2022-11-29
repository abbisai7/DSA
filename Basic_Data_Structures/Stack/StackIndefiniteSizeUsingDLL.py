class Node:
    def __init__(self,data=None,prev=None,next=None):
        self.data = data
        self.prev = prev
        self.next = next

class DoubleLinkedList:
    
    #dummynode and cur 
    dummyNode = cur = Node(-1,None,None)

    def push(self,val):
        #create new node and set prev to cur
        node = Node(val,self.cur,None)
        #set cur's next to new node 
        self.cur.next = node
        #make cur as new node
        self.cur = node

        return

    def pop(self):
        #if no elements are present just return
        if self.dummyNode.next == None:
            return -1

        temp_data = self.cur.data
        temp = self.cur.prev
        #cur's prev next to none
        self.cur.prev.next = None
        #set cur's prev to none
        self.cur.prev = None
        #make temp as cur
        self.cur = temp

        return temp_data

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
    stack = DoubleLinkedList()
    print(stack.printDLL())
    print(stack.pop())
    print(stack.push(1))
    print(stack.push(2))
    print(stack.push(3))
    print(stack.pop())
    print(stack.peek())
    print(stack.push(4))
    print(stack.printDLL())



