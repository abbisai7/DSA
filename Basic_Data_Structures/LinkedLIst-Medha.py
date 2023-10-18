class Node:
    def __init__(self,data,next=None):
        self.data = data
        self.next = next

#3(head)->1->2->X
#3->1(head)->2->X
#3-
#3->1(head)->2->X
#1(head)
    

#3->1(head)->2->X



#subba reddy(head)->umapathi reddy->milinda-->medha->x

class LinkedList:

    def __init__(self):
        self.head = None

    #insert at beginning
    #TC -> O(1)
    def insert_at_beginning(self,data):
        node = Node(data,self.head)
        self.head = node


    #insert at end
    #TC - O(L)
    def insert_at_end(self,data):
        if self.head == None:
            node = Node(data,self.head)
            self.head = node
            return
        
        temp = self.head
        while temp.next !=None:
            temp  = temp.next
        node =  Node(data)
        temp.next = node

    #get length
    #TC - O(L)
    def get_length(self):
        count = 0
        temp  = self.head

        while temp:
            count += 1
            temp = temp.next
        
        return count

    #print
    #TC - O(L)
    def print(self):
        res = ""
        temp = self.head

        while temp:
            res += str(temp.data)+"->"
            temp = temp.next
        
        return res

    


if __name__ == "__main__":
    linkedList = LinkedList()
    linkedList.insert_at_end("subba reddy")
    linkedList.insert_at_end("umapathi reddy")
    linkedList.insert_at_end("milli")
    print(linkedList.print())
    print(linkedList.get_length())
    linkedList.insert_at_end("medha")
    print(linkedList.get_length())
    print(linkedList.print())
