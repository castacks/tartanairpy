
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor

TYPE_OCV_2_TORCH_MAP = {
    np.uint8: torch.int32,
    np.dtype('uint8'): torch.int32,
    np.float32: torch.float32,
    np.dtype('float32'): torch.float32
}

TYPE_TORCH_2_OCV_MAP = {
    torch.int32: np.uint8,
    torch.int64: np.uint8,
    torch.float32: np.float32,
    torch.float64: np.float32,
}

def ocv_2_torch(img, keep_dtype=False):
    '''
    If keep_type is True, the returned tensor will have a appropriate torch dtype.
    And the value of the tensor will not be normalized to [0.0, 1.0].
    
    The returned tensor will have the batch dimension. That is [1, C, H, W]
    '''
    
    if keep_dtype:
        t = torch.from_numpy(img).to(dtype=TYPE_OCV_2_TORCH_MAP[img.dtype])
        
        # Make sure we have the appropirate shape.
        if t.ndim == 2:
            t = t.unsqueeze(0)
            
        t = t.permute((2, 0, 1))
    else:
        t = to_tensor(img)
        
    return t.unsqueeze(0)

def torch_2_ocv(img, scale=True, dtype=None):
    '''
    If scale=True, then the values are first scaled by 255.0.
    If dtype is np.uint8, the values in img are first clamped to [0.0, 255.0].
    If dtype is None, then the dtype of the output will be determined by TYPE_MAP.
    '''
    
    if scale:
        img = img * 255.0
    
    if dtype is None:
        dtype = TYPE_TORCH_2_OCV_MAP[img.dtype]
        
    if dtype == np.uint8:
        img = img.clamp(0.0, 255.0)
        
    if img.ndim == 4:
        img = img.permute((0, 2, 3, 1))
        
        if img.shape[0] == 1:
            img = img.squeeze(0).squeeze(-1) if img.shape[-1] == 1 else img.squeeze(0)
            return img.cpu().numpy().astype(dtype)
        else:
            img = img.squeeze(-1) if img.shape[-1] == 1 else img
            array = img.cpu().numpy().astype(dtype)
            return [ np.squeeze(a, axis=0) for a in np.split( array, img.shape[0], axis=0 ) ]
    else:
        img = img.permute((1, 2, 0))
        img = img.squeeze(-1) if img.shape[-1] == 1 else img
        return img.cpu().numpy().astype(dtype)
        
if __name__ == '__main__':
    # Test ocv_2_torch with uint8 type.
    print('Test ocv_2_torch with uint8 type.')
    img_ocv = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)
    print(f'img_ocv = \n{img_ocv}')
    img_torch = ocv_2_torch(img_ocv)
    print(f'img_torch.shape = {img_torch.shape}')
    print(f'img_torch.dtype = {img_torch.dtype}')
    print(f'img_torch = \n{img_torch}')
    
    img_ocv = torch_2_ocv(img_torch)
    print(f'img_ocv.shape = {img_ocv.shape}')
    print(f'img_ocv.dtype = {img_ocv.dtype}')
    print(f'img_ocv = \n{img_ocv}')
    
    # Test ocv_2_torch with uint8 type and keep_dtype=True.
    print('')
    print('Test ocv_2_torch with uint8 type and keep_dtype=True.')
    img_ocv = (np.random.rand(4, 4, 3) * 255).astype(np.uint8)
    print(f'img_ocv = \n{img_ocv}')
    img_torch = ocv_2_torch(img_ocv, keep_dtype=True)
    print(f'img_torch.shape = {img_torch.shape}')
    print(f'img_torch.dtype = {img_torch.dtype}')
    print(f'img_torch = \n{img_torch}')
    
    img_ocv = torch_2_ocv(img_torch, scale=False)
    print(f'img_ocv.shape = {img_ocv.shape}')
    print(f'img_ocv.dtype = {img_ocv.dtype}')
    print(f'img_ocv = \n{img_ocv}')
    
    # Test torch_2_ocv with batch size > 1.
    print('')
    print('Test torch_2_ocv with batch size > 1.')
    img_torch = torch.randint(0, 255, (2, 3, 4, 4))
    imgs_ocv = torch_2_ocv(img_torch, scale=False)
    for img in imgs_ocv:
        print(f'img.shape = {img.shape}')
        
    # Test torch_2_ocv with batch size = 1.
    print('')
    print('Test torch_2_ocv with batch size = 1.')
    img_torch = torch.randint(0, 255, (1, 3, 4, 4))
    print(f'img_torch = \n{img_torch}')
    img_ocv = torch_2_ocv(img_torch, scale=False)
    print(f'img_ocv.shape = {img_ocv.shape}')
    print(f'img_ocv = \n{img_ocv}')
    