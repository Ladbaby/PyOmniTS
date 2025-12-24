# define some globally shared, fixed information for all scripts to lookup
get_dataset_info() {
    local dataset_name=$1
    local dataset_subset_name=$2
    
    case $dataset_name in
        P12)
            dataset_root_path="storage/datasets/P12"
            n_variables=36
            ;;
        ECL)
            dataset_root_path="storage/datasets/electricity"
            n_variables=321
            ;;
        ETTh1|ETTm1)
            dataset_root_path="storage/datasets/ETT-small"
            n_variables=7
            ;;
        HumanActivity)
            dataset_root_path="storage/datasets/HumanActivity"
            n_variables=12
            ;;
        ILI)
            dataset_root_path="storage/datasets/illness"
            n_variables=7
            ;;
        MIMIC_III)
            dataset_root_path="storage/datasets/MIMIC_III"
            n_variables=96
            ;;
        MIMIC_IV)
            dataset_root_path="storage/datasets/MIMIC_IV"
            n_variables=100
            ;;
        P12)
            dataset_root_path="storage/datasets/P12"
            n_variables=36
            ;;
        Traffic)
            dataset_root_path="storage/datasets/traffic"
            n_variables=862
            ;;
        USHCN)
            dataset_root_path="storage/datasets/USHCN"
            n_variables=5
            ;;
        Weather)
            dataset_root_path="storage/datasets/weather"
            n_variables=21
            ;;
        *)
            echo "Error: Dataset '$dataset_name' not found in scripts/globals.sh" >&2
            return 1
            ;;
    esac
    
    return 0
}