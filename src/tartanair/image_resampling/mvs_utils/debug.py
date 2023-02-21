
import inspect

import torch

_ENABLED = True

def enable():
    global _ENABLED
    _ENABLED = True
    
def disable():
    global _ENABLED
    _ENABLED = False
    
def is_enabled():
    global _ENABLED
    return _ENABLED

def is_disabled():
    global _ENABLED
    return not _ENABLED

def this_line():
    frame_info = inspect.stack()[1]
    return f'{frame_info.filename}:{frame_info.lineno}'

def caller_line():
    frame_info = inspect.stack()[2]
    return f'{frame_info.filename}:{frame_info.lineno}'

def show_msg(msg):
    if is_disabled():
        return
    str_caller_line = caller_line()
    print(f'\n>>> DEBUG >>> {str_caller_line}: \n{msg}')

def show_obj(**kwargs):
    if is_disabled():
        return
    str_caller_line = caller_line()
    print(f'\n>>> DEBUG >>> {str_caller_line}: objects: ')
    for key, value in kwargs.items():
        print(f'{key}: {value}')

def show_sum(**kwargs):
    if is_disabled():
        return
    str_caller_line = caller_line()
    
    # Get the sum of all the inputs.
    print(f'\n>>> DEBUG >>> {str_caller_line}: sum of objects: ')
    for key, value in kwargs.items():
        print(f'{key}: {torch.sum(value)}')
        
def show_elements(indices, **kwargs):
    if is_disabled():
        return
    str_caller_line = caller_line()
    
    print(f'\n>>> DEBUG >>> {str_caller_line}: element of objects: ')
    for key, value in kwargs.items():
        print(f'{key}: ', end='')
        for elem in value.view((-1))[indices]:
            print(f'{elem}, ', end='')
        print()

def save_tensor(fn, **kwargs):
    if is_disabled():
        return
    str_caller_line = caller_line()
    
    print(f'\n>>> DEBUG >>> {str_caller_line}: save tensors: ')
    names = [ key for key in kwargs.keys() ]
    torch.save( kwargs, fn )
    print(f'{names} saved to {fn}')
    