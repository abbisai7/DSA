import re


class HashTable:

    def __init__(self):
        self.MAX = 100
        self.arr = [[] for x in range(self.MAX)]

    def get_hash(self,key):
        h = 0
        for i in key:
            h += ord(i)
        return h%self.MAX

    def __setitem__(self, key, name):
        x = self.get_hash(key)
        found = False
        for idx, elem in enumerate(self.arr[x]):
            if len(elem) == 2 and elem[0] == key:
                self.arr[x][idx] = (key,name)
                found = True
                break
        if not found:
            self.arr[x].append((key,name))

    def __getitem__(self, key):
        x = self.get_hash(key)
        for element in self.arr[x]:
            if element[0] == key:
                return element[1]


    def __delitem__(self, key):
        x = self.get_hash(key)
        for idx, element in enumerate(self.arr[x]):
            if element [0] == key:
                del self.arr[x][idx]
        
if __name__ == "__main__":
    t = HashTable()
    t["march 6"] = 20
    t["march 15"] = 21
    t["march 14"] = 22
    t["march 17"] = 23

    print(t.arr)
    print(t["march 14"])
    print(t["march 17"])
    del t["march 6"]