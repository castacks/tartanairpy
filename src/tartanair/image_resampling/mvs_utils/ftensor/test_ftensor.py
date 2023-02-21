
# Author
# ======
# 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# 
# Data
# ====
#
# Date: 2022-07-14
#

import argparse
import inspect
import unittest

import numpy as np

import torch

from .ftensor import ( FTensor, f_eye, f_zeros, f_ones )

def show_func_name():
    return inspect.stack()[1][3]

def print_delimiter(banner, n=10, d='='):
    print('\n' + d * n + ' ' + banner + ' ' + d * n + '\n')

def all_close_ft(a, b, rtol=1e-4, atol=1e-6):
    
    if isinstance(a, FTensor) and isinstance(b, FTensor):
        if a.f0 != b.f0 or a.f1 != b.f1:
            return False
        
        if a._is_rotation != b._is_rotation:
            return False
    
    return torch.allclose( a, b, rtol=rtol, atol=atol )

class TestFTensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        
        # point_array = torch.eye(3, dtype=torch.float32).repeat(1,2)
        point_array = torch.cat( 
                    ( torch.eye(3, dtype=torch.float32), 
                      torch.rand((3, 3), dtype=torch.float32) ), axis=1 )
        cls.point_array = FTensor(
            point_array.clone(), 
            f0='cif')
        
        # print(f'point_array = \n{point_array}')
        # print(f'cls.point_array = \n{cls.point_array}')
        # print(f'cls.point_array.numpy() = \n{cls.point_array.numpy()}')
        
        R_cbf_cif = torch.zeros((3,3), dtype=torch.float32)
        R_cbf_cif[1, 0] = 1
        R_cbf_cif[2, 1] = 1
        R_cbf_cif[0, 2] = 1
        cls.R_cbf_cif = FTensor(
            R_cbf_cif.clone(),
            f0='cbf', f1='cif',
            rotation=True)
        
        # print(f'R_cbf_cif = \n{R_cbf_cif}')
        # print(f'cls.R_cbf_cif = {cls.R_cbf_cif}')
        
        point_cbf = R_cbf_cif @ point_array
        cls.point_cbf = FTensor(
            point_cbf.clone(),
            f0='cbf')
        
        print(f'point_cbf = \n{point_cbf}')
        print(f'cls.point_cbf = {cls.point_cbf}')
        
        point_cbf_cuda = R_cbf_cif.to(device='cuda') @ point_array.to(device='cuda')
        print(f'point_cbf_cuda = \n{point_cbf_cuda}')
        
        R_rbf_cbf = torch.zeros((3,3), dtype=torch.float32)
        R_rbf_cbf[2, 0] = -1
        R_rbf_cbf[1, 1] =  1
        R_rbf_cbf[0, 2] =  1
        cls.R_rbf_cbf = FTensor(
            R_rbf_cbf.clone(),
            f0='rbf', f1='cbf',
            rotation=True)
        
        point_rbf = R_rbf_cbf @ point_cbf
        cls.point_rbf = FTensor(
            point_rbf.clone(),
            f0='rbf')

    def test_numpy(self):
        print_delimiter( show_func_name() )
        
        point_array = TestFTensor.point_array.numpy()
        R_cbf_cif   = TestFTensor.R_cbf_cif.numpy()
        
        point_cbf = R_cbf_cif @ point_array
        
        print(f'point_array = \n{point_array}')
        print(f'R_cbf_cif = \n{R_cbf_cif}')
        print(f'point_cbf = \n{point_cbf}')
    
    def test_clone(self):
        print_delimiter( show_func_name() )
        
        point_array_cloned = TestFTensor.point_array.clone()
        print(f'point_array_cloned = \n{point_array_cloned}')
            
        test_entries = [
            {'device': 'cpu'},
            {'device': 'cuda'},
        ]
        
        for entry in test_entries:
            print(entry)
            device = entry['device']
            
            point_array_device = TestFTensor.point_array.to(device=device)
            print(f'point_array_device = \n{point_array_device}')
    
        for entry in test_entries:
            print(entry)
            device = entry['device']
            
            point_array_cd = TestFTensor.point_array.clone().to(device=device)
            print(f'point_array_cd = \n{point_array_cd}')

    def test_basic_mutations(self):
        print_delimiter( show_func_name() )
        
        test_entries = [
            {'device': 'cpu'}, 
            {'device': 'cuda'}
        ]
        
        for entry in test_entries:
            print(entry)
            device = entry['device']
            
            print(f'Clone and convert the raw FTensor to device: {device}. ')
            ft = self.point_array.clone().to(device=device)
            # ft = TestFTensor.point_array.clone()
            print(f'The new FTensor is \n{ft}')
            
            print()
            print(f'Convert the dtype to torch.int: ')
            ft_int = ft.to(dtype=torch.int)
            print(f'ft_int = {ft_int}')
            print()
    
    def test_transform_point(self):
        print_delimiter( show_func_name() )
        
        test_entries = [
            {'device': 'cpu'},
            {'device': 'cuda'},
        ]
        
        for entry in test_entries:
            print(entry)
            
            device = entry['device']
            
            # Get a copy of the point_array and the rotation matrices.
            point_array = TestFTensor.point_array.clone().to(device=device)
            R_cbf_cif   = TestFTensor.R_cbf_cif.clone().to(device=device)
            R_rbf_cbf   = TestFTensor.R_rbf_cbf.clone().to(device=device)
            
            # print(f'point_array = \n{point_array}')
            # print(f'R_cbf_cif = \n{R_cbf_cif}')
            # print(f'R_rbf_cbf = \n{R_rbf_cbf}')
            
            # Transfer the true values to the device.
            true_point_cbf = TestFTensor.point_cbf.clone().to(device=device)
            true_point_rbf = TestFTensor.point_rbf.clone().to(device=device)
            
            # Transform the point_array from cif to cbf.
            point_cbf = R_cbf_cif @ point_array
            print(f'point_cbf = {point_cbf}\ntrue_point_cbf = {true_point_cbf}')
            
            self.assertTrue( all_close_ft(point_cbf, true_point_cbf), f'{show_func_name()} failed with entry {entry}' )
            
            # Transform the point_array from cif to rbf.
            point_rbf = R_rbf_cbf @ R_cbf_cif @ point_array
            print(f'point_rbf = {point_rbf}\ntrue_point_rbf = {true_point_rbf}')
            
            self.assertTrue( all_close_ft(point_rbf, true_point_rbf), f'{show_func_name()} failed with entry {entry}' )
    
    def test_arithmatics(self):
        print_delimiter( show_func_name() )
        
        t = TestFTensor.point_array.clone().tensor()
        f0 = TestFTensor.point_array.f0
        f1 = TestFTensor.point_array.f1
        
        # The raw true data.
        raw_true_add   = FTensor(t + t, f0=f0, f1=f1)
        raw_true_sub   = FTensor(t - t, f0=f0, f1=f1)
        raw_true_mul   = FTensor(t * t, f0=f0, f1=f1)
        raw_true_add_1 = FTensor(t + 1, f0=f0, f1=f1)
        raw_true_1_add = FTensor(1 + t, f0=f0, f1=f1)
        raw_true_sub_1 = FTensor(t - 1, f0=f0, f1=f1)
        raw_true_1_sub = FTensor(1 - t, f0=f0, f1=f1)
        raw_true_mul_2 = FTensor(t * 2, f0=f0, f1=f1)
        raw_true_2_mul = FTensor(2 * t, f0=f0, f1=f1)
        
        test_entries = [
            {'device': 'cpu'},
            {'device': 'cuda'},
        ]
        
        for entry in test_entries:
            print(entry)
            
            device = entry['device']
            
            # Get a copy of the point_array.
            ft = TestFTensor.point_array.clone().to(device=device)
            
            # Test additino between two FTensors.
            true_add = raw_true_add.to(device=device)
            added_two_ft = ft + ft
            self.assertTrue( all_close_ft( added_two_ft, true_add ), f'{show_func_name()} failed with entry {entry}' )
            
            # Test subtraction between two FTensors.
            true_sub = raw_true_sub.to(device=device)
            subtracted_two_ft = ft - ft
            self.assertTrue( all_close_ft( subtracted_two_ft, true_sub ), f'{show_func_name()} failed with entry {entry}' )
            
            # Test multiplication between two FTensors.
            try:
                multiplied_two_ft = ft * ft
                self.assertTrue( False, f'{show_func_name()} should not be able to multiply two FTensors. ' )
            except Exception as exc:
                print(f'Exception caught: \n{exc}')
                
            true_mul = raw_true_mul.to(device=device)
            multiplied_two_ft = ft.tensor() * ft
            self.assertTrue( all_close_ft( multiplied_two_ft, true_mul ), f'{show_func_name()} failed with entry {entry}' )
            
            # Test addition with 1.
            true_add_1 = raw_true_add_1.to(device=device)
            true_1_add = raw_true_1_add.to(device=device)
            added_ft_1 = ft + 1
            added_1_ft = 1 + ft
            self.assertTrue( all_close_ft( added_ft_1, true_add_1 ), f'{show_func_name()} failed with entry {entry}' )
            self.assertTrue( all_close_ft( added_1_ft, true_1_add ), f'{show_func_name()} failed with entry {entry}' )
            
            # Test subtraction with 1.
            true_sub_1 = raw_true_sub_1.to(device=device)
            true_1_sub = raw_true_1_sub.to(device=device)
            subtracted_ft_1 = ft - 1
            subtracted_1_ft = 1 - ft
            self.assertTrue( all_close_ft( subtracted_ft_1, true_sub_1 ), f'{show_func_name()} failed with entry {entry}' )
            self.assertTrue( all_close_ft( subtracted_1_ft, true_1_sub ), f'{show_func_name()} failed with entry {entry}' )
            
            # Test multiplication with 2.
            true_mul_2 = raw_true_mul_2.to(device=device)
            true_2_mul = raw_true_2_mul.to(device=device)
            multiplied_ft_2 = ft * 2
            multiplied_2_ft = 2 * ft
            self.assertTrue( all_close_ft( multiplied_ft_2, true_mul_2 ), f'{show_func_name()} failed with entry {entry}' )
            self.assertTrue( all_close_ft( multiplied_2_ft, true_2_mul ), f'{show_func_name()} failed with entry {entry}' )
            
            # Test adding two FTensors with inconsistent frames.
            ft2 = ft.clone()
            ft2.f0 = 'random'
            
            try:
                ft2 = ft2 + ft
                self.assertTrue( False, f'{show_func_name()} should not be able to add two FTensors with inconsistent frames. ' )
            except Exception as exc:
                print(f'Exception caught: \n{exc}')
                
            try:
                ft2 = ft + ft2
                self.assertTrue( False, f'{show_func_name()} should not be able to add two FTensors with inconsistent frames. ' )
            except Exception as exc:
                print(f'Exception caught: \n{exc}')
    
    def test_usual_torch_functions(self):
        print_delimiter( show_func_name() )
        
        test_entries = [
            {'device': 'cpu'},
            {'device': 'cuda'},
        ]
        
        for entry in test_entries:
            print(entry)
            
            device = entry['device']
            
            ft = TestFTensor.point_array.clone().to(device=device)
            
            # Test slicing.
            sliced = ft[:, 3:]
            print(f'sliced = {sliced}')
            
            # Test concatenation.
            cat_t = torch.cat( (ft, ft), axis=1 )
            print(f'cat_t = {cat_t}')
            self.assertTrue( isinstance( cat_t, FTensor ) )
            
            cat_tt = torch.cat( (ft.tensor(), ft), axis=1 )
            print(f'cat_tt = {cat_tt}')
            self.assertTrue( isinstance( cat_t, FTensor ) )
            
    def test_autograd(self):
        print_delimiter( show_func_name() )
        
        raw_true_grad = torch.ones_like( TestFTensor.point_array.tensor() ) * 2
        
        test_entries = [
            {'device': 'cpu'},
            {'device': 'cuda'},
        ]
        
        for entry in test_entries:
            print(entry)
            
            device = entry['device']
            
            true_grad = raw_true_grad.to(device=device)
            
            # Get the point_array.
            ft = TestFTensor.point_array.clone().to(device=device)
            ft.requires_grad = True
            
            # Some simple computation.
            ft2 = ft * 2
            ft2.backward( torch.ones_like( ft.tensor() ) )
            
            # Show the gradients.
            print(f'ft.grad = {ft.grad}')
            self.assertTrue( all_close_ft(ft.grad, true_grad), f'{show_func_name()} failed with entry {entry}' )
            
            # Get another point_array copy.
            ft = TestFTensor.point_array.clone().to(device=device)
            ft.requires_grad = True
            
            # Some simple computation.
            ft3 = ft * 2
            ft3 = ft3.tensor()
            ft3.backward( torch.ones_like( ft.tensor() ) )
            
            print(f'ft3: ft.grad = {ft.grad}')
            
            self.assertTrue( all_close_ft(ft.grad, true_grad), f'{show_func_name()} failed with entry {entry}' )
            
            # Get the tensor version of ft.
            t4 = TestFTensor.point_array.detach().tensor().clone().to(device=device)
            t4.requires_grad = True
            ft5 = FTensor(t4, f0=ft.f0, f1=ft.f1)
            ft6 = ft5 * 2
            ft6 = ft6.tensor()
            ft6.backward( torch.ones_like( ft.tensor() ) )
            
            print(f'ft6: t4.grad = \n{t4.grad}')
            
            self.assertTrue( all_close_ft(t4.grad, true_grad), f'{show_func_name()} failed with entry {entry}' )
    
    def test_inverse(self):
        print_delimiter( show_func_name() )

        raw_true_inv = FTensor( 
                    TestFTensor.R_cbf_cif.tensor().transpose(0, 1),
                    f0=TestFTensor.R_cbf_cif.f1,
                    f1=TestFTensor.R_cbf_cif.f0,
                    rotation=True)

        test_entries = [
            {'device': 'cpu'},
            {'device': 'cuda'},
        ]
        
        for entry in test_entries:
            print(entry)
            
            device = entry['device']
            
            # Get the true value.
            true_inv = raw_true_inv.to(device=device)
            
            # Get the Rotation matrix.
            R_cbf_cif = TestFTensor.R_cbf_cif.clone().to(device=device)
            
            # Inverse.
            R_cif_cbf = R_cbf_cif.inverse()
            
            self.assertTrue( all_close_ft( R_cif_cbf, true_inv ), f'{show_func_name()} failed with entry {entry}' )
    
    def test_transform_by_addition_only(self):
        print_delimiter( show_func_name() )
        
        raw_tr = FTensor( torch.rand((3,1)), f0='cbf' )
        raw_true_t = FTensor( 
                TestFTensor.point_array.tensor() + raw_tr.tensor(),
                f0='cbf' )
        
        test_entries = [
            {'device': 'cpu'},
            {'device': 'cuda'},
        ]
        
        for entry in test_entries:
            print(entry)
            
            device = entry['device']
            
            tr = raw_tr.to(device=device)
            true_t = raw_true_t.to(device=device)
            ft = TestFTensor.point_array.clone().to(device=device)
            
            # With the current definition, a vector cannot be transformed by addition only.
            # # Transform.
            # t = tr + ft
            # self.assertTrue( all_close_ft( t, true_t ), f'{show_func_name()} failed with entry {entry}' )
            
            # Invalid transform.
            try:
                t = ft + tr
                self.assertTrue( False, f'{show_func_name()} vector addition is not commutative between different frames. entry: {entry}' )
            except Exception as exc:
                print(f'Exception caught: \n{exc}')
                
    def test_tranformation_matrix(self):
        print_delimiter( show_func_name() )
        
        T_tensor = torch.eye(4)
        T_tensor[:3, :3] = TestFTensor.R_cbf_cif.tensor()
        T_tensor[:3,  3] = torch.rand((3,))
        
        i_tensor = torch.eye(4)
        i_tensor[:3, :3] = T_tensor[:3, :3].transpose(0, 1)
        i_tensor[:3,  3] = -i_tensor[:3, :3] @ T_tensor[:3, 3]
        
        T = FTensor( T_tensor.clone(), f0='cbf', f1='cif' )
        true_inv = FTensor( i_tensor, f0='cif', f1='cbf' )
        print(f'T = {T}')
        print(f'true_inv = {true_inv}')
        
        Tc = T.clone()
        Tc[:3, :3] = T[:3, :3]
        print(f'Tc = {Tc}')
        
        Tc = T.clone()
        Tc.f0 = T.f1
        Tc.f1 = T.f0
        
        with self.assertRaises(Exception, msg=f'{show_func_name()} failed to reise an Exception. '):
            Tc[:3, :3] = T[:3, :3]
            print(f'Tc = {Tc}')
        
        Ti = T.inverse()
        print(f'Ti = {Ti}')
        self.assertTrue( all_close_ft( Ti, true_inv ), f'{show_func_name()} failed when check inverse. ' )
    
    def test_constructors(self):
        print_delimiter( show_func_name() )
        
        raw_true_eye   = torch.eye(4, dtype=torch.float32)
        raw_true_zeros = torch.zeros((3, 3), dtype=torch.float32)
        raw_true_ones  = torch.ones((3, 3), dtype=torch.float32)
        
        test_entries = [
            {'device': 'cpu'},
            {'device': 'cuda'},
        ]
        
        for entry in test_entries:
            print(entry)
            
            device = entry['device']
            
            # f_eye.
            ft_eye = f_eye(4, 'rig', 'rig', dtype=torch.float32, device=device)
            true_eye = raw_true_eye.to(device=device)
            self.assertTrue( all_close_ft( ft_eye, true_eye ), f'{show_func_name()} failed when testing f_eye. entry {entry}' )
            
            # f_zeros.
            ft_zeros = f_zeros((3, 3), 'rig', 'rig', dtype=torch.float32, device=device)
            true_zeros = raw_true_zeros.to(device=device)
            self.assertTrue( all_close_ft( ft_zeros, true_zeros ), f'{show_func_name()} failed when testing f_zeros. entry {entry}' )
            
            # f_ones.
            ft_ones = f_ones((3, 3), 'rig', 'rig', dtype=torch.float32, device=device)
            true_ones = raw_true_ones.to(device=device)
            self.assertTrue( all_close_ft( ft_ones, true_ones ), f'{show_func_name()} failed when testing f_ones. entry {entry}' )
    
if __name__ == '__main__':
    import sys
    
    parser = argparse.ArgumentParser(description='Test the FTensor class.')
    
    parser.add_argument('--disable-amp', action='store_true', default=False, 
                        help='Disable automatic mixed precision for matrix multiplication.')
    
    args = parser.parse_args()
    
    if args.disable_amp:
        torch.backends.cuda.matmul.allow_tf32 = False
    
    # Make sure unittest is happy.
    sys.argv = sys.argv[:1]
    
    unittest.main()