class Node:
    def __init__(self,data=None,prev=None,next=None):
        self.data = data
        self.prev = prev
        self.next = next

class DoubleLinkedList:
    
    dummyNode = cur = Node(-1,None,None)

    def push(self,val):
        node = Node(val,self.cur,None)
        self.cur.next = node
        self.cur = node

        return

    def pop(self):
        if self.dummyNode.next == None:
            return -1

        temp = self.cur.prev
        self.cur.prev.next = None
        self.cur.prev = None
        self.cur = temp

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



