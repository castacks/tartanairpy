import os
from os.path import join, isfile, basename
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Local imports.
from .tartanair_module import TartanAirModule, print_error, print_highlight, print_warn


class Unzipper(TartanAirModule):
    def __init__(self, tartanair_data_root):
        super().__init__(tartanair_data_root)

    # ────────────────────────────────
    # Discover zip files
    # ────────────────────────────────

    def collect_zip_files(self):
        zipfiles = []
        for root, _, files in os.walk(self.tartanair_data_root):
            rel_root = os.path.relpath(root, self.tartanair_data_root)
            for file in files:
                if file.endswith('.zip'):
                    zipfiles.append(join(rel_root, file))
        return zipfiles
    
    def unzip_files(self, zipfile, output_dir, remove_after = False):
        if not isfile(zipfile) or (not zipfile.endswith('.zip')):
            print_error("The zip file is missing {}".format(zipfile))
            return False
        print('  Unzipping {} ...'.format(zipfile))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print_warn("  Created output directory: {}".format(output_dir))
            
        cmd = 'unzip -q -o ' + zipfile + ' -d ' + output_dir
        os.system(cmd)

        if remove_after:
            cmd = 'rm ' + zipfile
            os.system(cmd)
            print('  Removed {}'.format(zipfile))

        print_highlight("  Unzipping Completed! ")

    def unzip_all(self, output_dir, num_workers):
        '''
        mode can be 'tartanair' or 'tartanground'
        '''
        print_warn("⚠️  Unzipping assumes the original file path structure from the downloading script...")
        print_warn("⚠️  Unzipping will overwrite existing files...")

        zipfiles = self.collect_zip_files()
        print(f"📦 Found {len(zipfiles)} zip files.")

        outdirs = []
        for zf in zipfiles:
            # TartanAir and TartanGround have different folder structures
            if zf.find('Data_easy') != -1 or zf.find('Data_hard') != -1: # hard coded for TartanAir
                outdirs.append(output_dir) 
            else:
                outdirs.append(join(output_dir, os.path.dirname(zf)) )

        zipfiles = [join(self.tartanair_data_root, zf) for zf in zipfiles]

        # for zf, od in tqdm(zip(zipfiles, outdirs), total=len(zipfiles), desc="Unzipping"):
        #     self.unzip_files(zf, od)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.unzip_files, zf, od): (zf, od) for zf, od in zip(zipfiles, outdirs)}
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Unzipping"):
                pass

        print_highlight("🎉 All unzips completed.")

