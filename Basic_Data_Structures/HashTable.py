

class HashTable:

    def __init__(self):
        self.MAX = 100
        self.arr = [None for i in range(self.MAX)]

    def get_hash(self,key):
        h = 0
        for i in key:
            h += ord(i)
        return h % self.MAX
    
    def __setitem__(self, key, value):
        x = self.get_hash(key)
        self.arr[x] = value
    
    def __getitem__(self,key):
        x = self.get_hash(key)
        return self.arr[x]

    def __delitem__(self, key):
        x = self.get_hash(key)
        self.arr[x] = None

if __name__ == "__main__":
    t = HashTable()
    t["march 16"] = 20
    t["march 15"] = 21
    t["march 14"] = 22

    print(t["march 14"])
    print(t.arr)
    del t["march 15"]
    print(t.arr)
    print(t["march 16"])

