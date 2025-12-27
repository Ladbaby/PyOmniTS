# Code from: https://github.com/Ladbaby/PyOmniTS
'''
Launcher script that invokes all MIMIC-III preprocessing scripts sequentially.

sha256 checksums of output files are also checked.
'''

import hashlib
import os
import subprocess
import sys
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
        "5_DataMerging.py"
    ]
    output_file_name_and_checksum_dict = {
        1: { # indexed by iteration counter
            "Admissions_processed.csv": "30a53660f3afc5a3e871a73315a8e82bdd72c21525716bef9a8fbff7e748268e",
            "INPUTS_processed.csv": "6b5a76e98e482b8cde5235028e20c9b7a47b43dbe6f3fa57171b043b892c1a22"
        },
        2: {
            "OUTPUTS_processed.csv": "c9147b708e484cd2a63f4544f713a078b8bc9cc921491d1601695c4a25806c5a"
        },
        3: {
            "LAB_processed.csv": "cfd08f9837e3cbea6fd876ad11bd6bb7c999431a8ad2c9146c659145a7b9359c"
        },
        4: {
            "PRESCRIPTIONS_processed.csv": "86ebf15ae2b69adeaddc6c65f554dd2557a673589096c784302dc0f41a8d8194"
        },
        5: {
            "complete_tensor.csv": "8106f64292771956f70ccd0ca1a4f7a0a4563fe63d6eff6ee2ef27dc6fdb614a"
        }
    }

    raw_file_path = input("Enter the folder path to MIMIC-III raw files (i.e., the folder path containing many .csv.gz files): ")
    if not raw_file_path.endswith('/'):
        raw_file_path += '/'
    raw_file_path = raw_file_path.replace("\\", '/')
    raw_file_path = os.path.expanduser(raw_file_path)
    outfile_path=os.path.expanduser("~/.tsdm/rawdata/MIMIC_III_DeBrouwer2019/")
    os.makedirs(outfile_path, exist_ok=True)

    for i, target_script_name in enumerate(target_script_name_list, start=1):
        target_script = os.path.join(script_dir, target_script_name)

        print(f"Running {target_script}")
        result = subprocess.run([sys.executable, target_script, raw_file_path]) # pass raw_file_path as command line argument
        for output_file_name, checksum_groundtruth in output_file_name_and_checksum_dict[i].items():
            if i == 5:
                checksum = sha256_checksum(outfile_path+output_file_name)
            else:
                checksum = sha256_checksum(raw_file_path+output_file_name)
            if checksum == checksum_groundtruth:
                print(f"Checksum verification for {output_file_name} passed.")
            else:
                print(f"WARNING: checksum incorrect for {output_file_name}!")

    print(f"""MIMIC-III data preprocessing finished! These intermediate files under {raw_file_path} can be deleted:
    - Admissions_processed.csv
    - INPUTS_processed.csv
    - OUTPUTS_processed.csv
    - LAB_processed.csv
    - PRESCRIPTIONS_processed.csv""")

if __name__ == "__main__":
    main()
