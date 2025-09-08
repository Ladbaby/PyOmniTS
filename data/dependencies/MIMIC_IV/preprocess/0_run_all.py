'''
Launcher script that invokes all MIMIC-IV preprocessing scripts sequentially.

sha256 checksums of output files are also checked.
'''

import os
import sys
import subprocess
import hashlib
import warnings

warnings.filterwarnings('ignore')

def sha256_checksum(filename, chunk_size=8192):
    sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def check_package_version():
    import numpy
    import pandas

    if numpy.__version__ != "1.21.6":
        print(f"WARNING: numpy==1.21.6 is strongly recommended. Current version is {numpy.__version__}. You may get incorrect results!")
    if pandas.__version__ != "1.3.5":
        print(f"WARNING: pandas==1.3.5 is strongly recommended. Current version is {pandas.__version__}. You may get incorrect results!")


def main():
    check_package_version()
    # Get the absolute path of the directory where this launcher script resides
    script_dir = os.path.dirname(os.path.abspath(__file__))

    target_script_name_list = [
        "1_Admissions.py",
        "2_Outputs.py",
        "3_LabEvents.py",
        "4_Prescriptions.py",
        "5_InputEvents.py",
        "6_DataMerging.py"
    ]
    output_file_name_and_checksum_dict = {
        1: { # indexed by iteration counter
            "Admissions_processed.csv": "07374d7fdfefd98b441c4c2708d98086b8b3e5df6be594732e8dc0f6ab4ca033"
        },
        2: {
            "OUTPUTS_processed.csv": "486ee10f69ddb65d6f9bea50a615134f20bfa64c73298717ff7736fda70a5c01"
        },
        3: {
            "LAB_processed.csv": "7b638a1ebf591502edcb106b7aaadf79a3f5075c9c81a477fb4d795a74d29780"
        },
        4: {
            "PRESCRIPTIONS_processed.csv": "fc62c4e13c28c1dc095327a32b6011c4af3da52b2114d9a2e82af8838bf13242"
        },
        5: {
            "INPUTS_processed.csv": "e9d58e237c42786fef9d73ac6dce80c3e1fbaad2614c5944e935da2854287e69"
        },
        6: {
            "full_dataset.csv": "cb90e0cef16d50011aaff7059e73d3f815657e10653a882f64f99003e64c70f5"
        }
    }

    raw_file_path = input("Enter the folder path to MIMIC-IV raw files (i.e., the folder path containing LICENSE.txt): ")
    if not raw_file_path.endswith('/'):
        raw_file_path += '/'
    raw_file_path = raw_file_path.replace("\\", '/')
    raw_file_path = os.path.expanduser(raw_file_path)
    outfile_path = os.path.expanduser("~/.tsdm/rawdata/MIMIC_IV_Bilos2021/")
    os.makedirs(outfile_path, exist_ok=True)

    for i, target_script_name in enumerate(target_script_name_list, start=1):
        target_script = os.path.join(script_dir, target_script_name)

        print(f"Running {target_script}")
        result = subprocess.run([sys.executable, target_script, raw_file_path]) # pass raw_file_path as command line argument
        for output_file_name, checksum_groundtruth in output_file_name_and_checksum_dict[i].items():
            if i == 6:
                checksum = sha256_checksum(outfile_path+output_file_name)
            else:
                checksum = sha256_checksum(f"{raw_file_path}processed/{output_file_name}")
            if checksum == checksum_groundtruth:
                print(f"Checksum verification for {output_file_name} passed.")
            else:
                print(f"WARNING: checksum incorrect for {output_file_name}!")

    print(f"""MIMIC-IV data preprocessing finished! These intermediate files under {raw_file_path}processed can be deleted:
    - Admissions_processed.csv
    - INPUTS_processed.csv
    - OUTPUTS_processed.csv
    - LAB_processed.csv
    - PRESCRIPTIONS_processed.csv""")

if __name__ == "__main__":
    main()
