from collections import deque
class Stack:
    def __init__(self):
        self.container = deque()

    def push(self, val):
        return self.container.append(val)
    
    def pop(self):
        return self.container.pop()

    def peek(self):
        return self.container[-1]

    def isEmpty(self):
        return len(self.container) == 0

    def size(self):
        return len(self.container)


if __name__ == "__main__":
    s = Stack()
    s.push(2)
    s.push(3)
    s.pop()
    s.push(3)
    print(s.isEmpty())
    print(s.size())
    print(s.peek())