class Node:
    def __init__(self, data = None, next = None):
        self.data = data
        self.next = next


class LinkedList:
    def __init__(self):
        self.head = None
    
    def insert_at_beginning(self,data):
        node = Node(data,self.head)
        #print(node)
        self.head = node

    def insert_at_end(self,data):
        if not self.head:
            node = Node(data)
            self.head = node
            return
        
        temp = self.head
        while temp.next:
            temp = temp.next
        temp.next = Node(data)

    def get_length(self):
        if self.head is None:
            print("Linked List is Empty")
            return

        temp = self.head
        count = 0
        while temp:
            count += 1
            temp = temp.next
        return count

    def insert_at(self, data, pos):
        if pos < 0 or pos > self.get_length():
            print("Invalid Index")

        if pos == 0 :
            self.insert_at_beginning(data)
            return

        temp = self.head
        count = 0
        while temp:
            if count == pos-1:
                node = Node(data,temp.next)
                temp.next = node
                break
            temp = temp.next
            count +=1

    def remove_at(self,pos):
        if pos < 0 or pos > self.get_length():
            print("Invalid Index")
            return
        
        if pos == 0:
            self.head = self.head.next
            return

        count = 0
        temp = self.head
        while temp:
            if count == pos-1:
                temp.next = temp.next.next
                break
                
            temp = temp.next
            count += 1

    def print(self):
        temp = self.head
        s = " "
        while temp:
            s += str(temp.data) + "-->" if temp.next else str(temp.data)
            temp = temp.next
        print(s)
    
    def insert_values(self,data_list):
        for data in data_list:
            self.insert_at_end(data)

    ##Searching a value
    def search(self,inp):
        temp = self.head
        count = 0
        pos = 0
        while temp:
            if temp.data == inp:
                pos = count
                break
            temp = temp.next
            count +=1

        if pos:
            print(pos)
        else:
            print("Not present in LL")

    def insert_before(self,before,data):
        temp = self.head
        preptr = temp
        if temp.data == before:
            self.insert_at_beginning(data)
            return

        while temp.data != before:
            preptr = temp
            temp = temp.next
        
        node = Node(data, preptr.next)
        preptr.next = node

    def insert_after(self,after,data):
        temp = self.head
        while temp.data != after:
            temp = temp.next
        
        node = Node(data,temp.next)
        temp.next = node

    def reverse_list(self):
        if self.head == None or self.head.next == None:
            print(self.head)
            return
        prev = None
        current = self.head

        while current:
            next = current.next#7
            current.next = prev
            prev = current
            current = next

        self.head = prev

    def sortList(self,head):

        if not head or not head.next:
            return head

        # split the list into two half's
        left = head
        right = self.getMid(head)
        temp = right.next
        right.next = None
        right = temp

        left = self.sortList(left)
        right = self.sortList(right)
        return self.merge(left,right)

    def getMid(self,head):
        slow = head
        fast = head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def merge(self, left, right):
        dummy_ptr = head_ptr = Node()
        while (left and right) :
            if left.data < right.data:
                dummy_ptr.next = left
                left = left.next
            else:
                dummy_ptr.next = right
                right = right.next
            dummy_ptr = dummy_ptr.next

        if left:
            dummy_ptr.next = left
        if right:
            dummy_ptr.next = right
        return head_ptr.next


    def addInBetween(self,n1,n2,val):
        node = Node(val)
        n1.next = node
        node.next = n2
        









        
    

if __name__ == '__main__':
    ll = LinkedList()
    ll.insert_at_beginning(2)
    ll.insert_at_beginning(3)
    ll.insert_at_beginning(4)
    ll.insert_at_end(-1)
    ll.insert_at_beginning(5)
    #ll.insert_at_end(1)
    #ll.insert_at_end(0)
    #ll.insert_at(-10,1)
    #ll.insert_values(["banana","mango","grapes","orange"])
    ll.print()
    #ll.remove_at(0)
    ll.get_length()
    #ll.print()
    #ll.insert_after(-10,6)
    ll.print()
    ll.reverse_list()
    ll.print()
    #print(ll.head)
    x=ll.sortList(ll.head)
    print(ll.head)
    ll.head = x
    ll.print()
