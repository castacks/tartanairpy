
class A(object):
    def __init__(self):
        self._data = 0
    
    @property
    def data(self):
        print('A: get data. ')
        return self._data
    
    @data.setter
    def data(self, value):
        print('A: set data. ')
        self._data = value
    
class B(A):
    def __init__(self):
        super().__init__()
        
    @A.data.getter
    def data(self):
        print('B: get data. ')
        return super().data
    
    @A.data.setter
    def data(self, value):
        print('B: set data. ')
        # super().data.__set__(self, value)
        A.data.fset(self, value)
        
class C(B):
    def __init__(self):
        super().__init__()
        
    @property
    def data(self):
        print('C: get data. ')
        return super().data
    
    @B.data.setter
    def data(self, value):
        print('C: set data. ')
        # super().data.__set__(self, value)
        B.data.fset(self, value)
        
if __name__ == '__main__':
    a = A()
    b = B()
    c = C()
    
    print(c.data) # Won't work as intented. 
    c.data = 1 # Works.
    