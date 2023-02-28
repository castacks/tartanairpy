'''
Author: Yorai Shaoul
Date: 2023-02-28

This file contains the lister class, which lists available environments locally and remotely.
'''
# General imports.
import os

from colorama import Fore, Style

# Local imports.
from .tartanair_module import TartanAirModule

class TartanAirLister(TartanAirModule):
    def __init__(self, tartanair_data_root):
        super().__init__(tartanair_data_root)

    def list_envs(self, verbose = True):
        '''
        List the available environments locally and remotely.
        '''
        # Get the local environments.
        local_envs = os.listdir(self.tartanair_data_root)

        # Get the remote environments.
        remote_envs = self.env_names

        # Print the results.
        if verbose:
            print(Fore.GREEN + 'Local environments:' + Style.RESET_ALL)
            for i, env in enumerate(local_envs):
                print("    ", i, ". ", env, sep="")
            print(Fore.GREEN + 'Remote environments:' + Style.RESET_ALL)
            for i, env in enumerate(remote_envs):
                print("    ", i, ". ", env, sep="") 

        # Return the results.
        return {'local': local_envs, 'remote': remote_envs}