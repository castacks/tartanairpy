
# Author
# ======
#
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
#
# Data
# ====
# 
# First:   2021-09-30
# Current: 2022-07-17
# 

import numpy as np

# from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion

class Printer(object):
    def __init__(self, indent='    '):
        super(Printer, self).__init__()

        self.indent = indent

    def indent_str(self, indent_level):
        s = ''
        for _ in range(indent_level):
            s += self.indent
        return s

    def make_str(self, key, value, indent_level=0):
        raise NotImplementedError

    def __str__(self, key, value, indent_level=0):
        return self.make_str(key, value, indent_level)

class PlainPrinter(Printer):
    def __init__(self):
        super(PlainPrinter, self).__init__()

    def make_str(self, key, value, indent_level=0):
        if ( isinstance(value, str) ):
            value_str = '\"{}\"'.format(value)
        elif ( isinstance(value, bool) ):
            value_str = 'true' if value == True else 'false'
        elif ( value is None ):
            value_str = '\"None\"'
        else:
            value_str = '{}'.format(value)
        
        if ( key is not None ):
            s = '{}\"{}\": {}'.format(self.indent_str(indent_level), key, value_str)
        else:
            s = '{}{}'.format(self.indent_str(indent_level), value_str)
        return s

class DictPrinter(Printer):
    def __init__(self):
        super(DictPrinter, self).__init__()

        self.opening = '{'
        self.ending = '}'

        self.printers = dict()

        self.str_printer = PlainPrinter()

    def __len__(self):
        return len(self.printers)

    def __getitem__(self, key):
        return self.printers[key]

    def __setitem__(self, key, printer):
        self.printers[key] = printer

    def keys(self):
        return self.printers.keys()

    def make_str(self, key, value, indent_level=0):
        assert( isinstance(value, (dict, PrettyDict)) ), \
            '{} must be a dict or PrettyDict. '.format(value)
        
        indent = self.indent_str(indent_level)
        if ( key is not None ):
            s = '%s\"%s\": %s\n' % (indent, key, self.opening)
        else:
            s = '%s%s\n' % ( indent, self.opening )

        indent_level += 1
        for k, v in value.items():
            if ( isinstance(v, str) ):
                s += '%s,\n' % ( self.str_printer.make_str(k, v, indent_level) )    
            else:
                p = self.printers[k]
                s += '%s,\n' % ( p.make_str(k, v, indent_level) )

        s = s[:-2]
        s += '\n'

        indent_level -= 1
        s += '%s%s' % (self.indent_str(indent_level), self.ending)
        return s

class ListPrinter(Printer):
    def __init__(self, linear=True):
        super(ListPrinter, self).__init__()
        self.linear = linear

        self.opening = '['
        self.ending = ']'

        self.printers = list()

        self.str_printer = PlainPrinter()

    def __len__(self):
        return len(self.printers)

    def __getitem__(self, index):
        return self.printers[index]

    def __setitem__(self, index, printer):
        self.printers[index] = printer

    def append(self, printer):
        self.printers.append(printer)

    def make_str(self, key, value, indent_level=0):
        assert( isinstance(value, (list, tuple)) ), \
            '{} must be a list or tuple. '.format(value)
        
        indent = self.indent_str(indent_level)

        sep = ' ' if self.linear else '\n'

        if ( key is not None ):
            s = '%s\"%s\": %s%s' % (indent, key, self.opening, sep)
        else:
            s = '%s%s%s' % ( indent, self.opening, sep )

        if ( self.linear ):
            indent_level_in_list = 0
        else:
            indent_level_in_list = indent_level + 1

        for i, v in enumerate(value):
            if ( isinstance(v, str) ):
                s += '%s,%s' % ( self.str_printer.make_str(None, v, indent_level_in_list), sep )    
            else:
                p = self.printers[i]
                s += '%s,%s' % ( p.make_str(None, v, indent_level_in_list), sep )

        s = s[:-2]
        s += sep

        if (self.linear):
            s += '%s' % (self.ending)
        else:
            s += '%s%s' % (self.indent_str(indent_level), self.ending)
        return s

class SequencePrinter(Printer):
    def __init__(self, have_indent=True):
        super(SequencePrinter, self).__init__()

        self.have_indent = have_indent

class NumPyPrinter(SequencePrinter):
    def __init__(self, have_indent=True):
        super(NumPyPrinter, self).__init__(have_indent)

    def make_str(self, key, value, indent_level=0):
        indent = self.indent_str(indent_level)
        s = ''
        if ( key is not None ):
            s += '{}\"{}\": {}'.format( indent, key, str(value.tolist()) )
        else:
            if ( self.have_indent ):
                s += '{}{}'.format(indent, str(value.tolist()) )
            else:
                s += '{}'.format( str(value.tolist()) )
        return s

class NumPyLineBreakPrinter(SequencePrinter):
    def __init__(self, shape, have_indent=True):
        super().__init__(have_indent)
        
        self.shape = shape
        
    def make_str(self, key, value, indent_level=0):
        # Reshape the array.
        value = value.reshape(self.shape)
        
        indent = self.indent_str(indent_level)
        s = ''
        if ( key is not None ):
            s += '{}\"{}\": [\n'.format( indent, key )
        else:
            if ( self.have_indent ):
                s += '{}[\n'.format(indent )
            else:
                s += '[\n'
                
        # Loop for all the rows.
        indent = self.indent_str(indent_level + 1)
        for i in range(self.shape[0]):
            s += '{}'.format(indent)
            for j in range(self.shape[1]):
                s += f'{value[i, j]}, '
            
            if i == self.shape[0] - 1:
                s = s[:-2]

            s += '\n'
            
        indent = self.indent_str(indent_level)
        s += f'{indent}]'
        
        return s

# class ROSPointPrinter(SequencePrinter):
#     def __init__(self, have_indent=True):
#         super(ROSPointPrinter, self).__init__(have_indent)

#     def make_str(self, key, value, indent_level=0):
#         value_list = [ value.x, value.y, value.z ]

#         indent = self.indent_str(indent_level)
#         s = ''
#         if ( key is not None ):
#             s += '{}\"{}\": {}'.format( indent, key, value_list )
#         else:
#             if ( self.have_indent ):
#                 s += '{}{}'.format(indent, value_list )
#             else:
#                 s += '{}'.format( value_list )
#         return s

# class ROSQuaternionPrinter(SequencePrinter):
#     def __init__(self, have_indent=True):
#         super(ROSQuaternionPrinter, self).__init__(have_indent)

#     def make_str(self, key, value, indent_level=0):
#         value_list = [ value.x, value.y, value.z, value.w ]

#         indent = self.indent_str(indent_level)
#         s = ''
#         if ( key is not None ):
#             s += '{}\"{}\": {}'.format( indent, key, value_list )
#         else:
#             if ( self.have_indent ):
#                 s += '{}{}'.format(indent, value_list )
#             else:
#                 s += '{}'.format( value_list )
#         return s

# class ROSPosePrinter(SequencePrinter):
#     def __init__(self, have_indent=True):
#         super(ROSPosePrinter, self).__init__(have_indent)

#     def make_str(self, key, value, indent_level=0):
#         value_list = [ 
#             [value.position.x, value.position.y, value.position.z ],
#             [value.orientation.x, value.orientation.y, value.orientation.z, value.orientation.w ]
#         ]

#         indent = self.indent_str(indent_level)
#         s = ''
#         if ( key is not None ):
#             s += '{}\"{}\": {}'.format( indent, key, value_list )
#         else:
#             if ( self.have_indent ):
#                 s += '{}{}'.format(indent, value_list )
#             else:
#                 s += '{}'.format( value_list )
#         return s

# class ROSPoseArrayPrinter(SequencePrinter):
#     def __init__(self, have_indent=True):
#         super(ROSPoseArrayPrinter, self).__init__(have_indent)
#         self.opening = '['
#         self.ending = ']'
#         self.pose_printer = ROSPosePrinter(have_indent)

#     def make_str(self, key, value, indent_level):
#         indent = self.indent_str(indent_level)

#         sep = '\n' if self.have_indent else ' '

#         if ( key is not None ):
#             s = '%s\"%s\": %s%s' % (indent, key, self.opening, sep)
#         else:
#             s = '%s%s%s' % ( indent, self.opening, sep )

#         if ( self.have_indent ):
#             indent_level_in_list = indent_level + 1
#         else:
#             indent_level_in_list = 0

#         for v in value.poses:
#             s += '%s,%s' % ( self.pose_printer.make_str(None, v, indent_level_in_list), sep )

#         s = s[:-2]
#         s += sep

#         if (self.have_indent):
#             s += '%s%s' % (self.indent_str(indent_level), self.ending)
#         else:
#             s += '%s' % (self.ending)
#         return s

def create_printer(value):
    if ( isinstance(value, str) ):
        return PlainPrinter()
    elif ( isinstance(value, np.ndarray) ):
        return NumPyPrinter()
    # elif ( isinstance(value, Point) ):
    #     return ROSPointPrinter()
    # elif ( isinstance(value, Quaternion) ):
    #     return ROSQuaternionPrinter()
    # elif ( isinstance(value, Pose) ):
    #     return ROSPosePrinter()
    # elif ( isinstance(value, PoseArray) ):
    #     return ROSPoseArrayPrinter()
    elif ( isinstance(value, dict) ):
        return create_printers_for_dict(value)
    elif ( isinstance(value, (list, tuple)) ):
        return create_printers_for_list(value)
    else:
        return PlainPrinter()

def create_printers_for_dict(in_dict):
    dp = DictPrinter()

    for key, value in in_dict.items():
        dp[key] = create_printer(value)
    
    return dp

def create_printers_for_list(in_list):
    # linear = False if isinstance(in_list[0], (list, dict, Pose, PoseArray, np.ndarray)) else True
    linear = False if isinstance(in_list[0], (list, dict, np.ndarray)) else True
    lp = ListPrinter(linear=linear)

    for entry in in_list:
        lp.append( create_printer(entry) )

    return lp

class PrettyDict(object):
    def __init__(self):
        super(PrettyDict, self).__init__()

        self.d = dict()
        self.p = DictPrinter()

        self.opening = '{'
        self.ending  = '}'

        self.str_printer = PlainPrinter()

    def __len__(self):
        return len(self.d)

    def __getitem__(self, key):
        return self.d[key]

    def __setitem__(self, key, value):
        self.d[key] = value

        if ( isinstance(value, PrettyDict) ):
            self.p[key] = value.p
        elif ( isinstance(value, np.ndarray) ):
            self.p[key] = NumPyPrinter()
        else:
            self.p[key] = PlainPrinter()

    def update(self, key, value, printer=PlainPrinter()):
        self.d[key] = value
        self.p[key] = printer

    def update_printer(self, key, printer):
        if ( not key in self.p.keys() ):
            raise KeyError

        self.p[key] = printer

    def auto_update_printer(self):
        for key, value in self.d.items():
            self.p[key] = create_printer(value)

    def items(self):
        return self.d.items()

    def make_str(self):
        s = '%s\n' % (self.opening)
        indent_level = 1

        for key, value in self.d.items():
            if ( isinstance(value, str) ):
                s += '%s,\n' % ( self.str_printer.make_str(key, value, indent_level) )
            else:
                p = self.p[key]
                s += '%s,\n' % ( p.make_str(key, value, indent_level) )

        s = s[:-2]
        s += '\n%s' % ( self.ending )
        return s

    def __str__(self):
        return self.make_str()

def test_simple_dict():
    print('Test simple dict.')
    pd = PrettyDict()
    pd['a'] = 'abc'
    pd['b'] = 'def'
    print(pd)
    print()

def test_simple_containers():
    print('Test simple containers.')
    pd = PrettyDict()
    pd['a'] = {
        'aa': 'abc',
        'ab': 'def'
    }

    dp = DictPrinter()
    dp['aa'] = PlainPrinter()
    dp['ab'] = PlainPrinter()
   
    pd.update_printer('a', dp)

    pd['b'] = [
        'abc', 'edf'
    ]

    lp = ListPrinter()
    lp.append(PlainPrinter())
    lp.append(PlainPrinter())

    pd.update_printer('b', lp)

    print(pd)
    print()

def test_mixed_dict():
    print('Test mixed dict.')
    pd0 = PrettyDict()
    pd0['a'] = {
        'aa': 'abc',
        'ab': 'def'
    }

    dp = DictPrinter()
    dp['aa'] = PlainPrinter()
    dp['ab'] = PlainPrinter()
   
    pd0.update_printer('a', dp)

    pd0['b'] = [
        'abc', 'edf'
    ]

    lp = ListPrinter()
    lp.append(PlainPrinter())
    lp.append(PlainPrinter())

    pd0.update_printer('b', lp)

    pd1 = PrettyDict()

    pd1['c'] = pd0
    pd1['d'] = {
        'e': 'eee',
        'f': 'fff'
    }

    print(pd1)
    print()

def test_numpy():
    print('Test NumPy. ')

    pd = PrettyDict()
    pd['a'] = [1, 2.0, 3]
    pd['b'] = np.array([1,2,3], dtype=np.float32)
    pd['c'] = [
        np.random.rand(3),
        np.random.rand(3)
    ]

    lp = ListPrinter(linear=False)
    lp.append(NumPyPrinter(have_indent=True))
    lp.append(NumPyPrinter(have_indent=True))

    pd.update_printer('c', lp)

    print(pd)
    print()

def test_auto_update():
    print('Test auto update.')

    pd = PrettyDict()
    pd['a'] = 1
    pd['b'] = 2.0
    pd['c'] = '3'
    pd['d'] = [1,2,3]
    pd['e'] = {
        'ea': 1,
        'eb': [1, 2, 3]
    }
    # pd['f'] = Point(10, 20, 30)
    # pd['g'] = Quaternion(0, 0, 0, 1)

    # pose = Pose()
    # pose.position = Point(10, 20, 30)
    # pose.orientation = Quaternion(0, 0, 0, 1)
    # pd['h'] = pose

    # pose_array = PoseArray()
    # pose_array.poses = [pose] * 3

    # pd['i'] = pose_array
    # pd['j'] = [pose_array] * 2

    array = np.random.rand(3)
    pd['k'] = array
    pd['l'] = [array] * 3

    pd.auto_update_printer()

    print(pd)
    print()

if __name__ == '__main__':
    test_simple_dict()
    test_simple_containers()
    test_mixed_dict()
    test_numpy()
    test_auto_update()