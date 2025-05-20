###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Authors: Yulia Rubanova and Ricky Chen
###########################

import os
import hashlib
import requests

import torch

# Adapted from: https://github.com/rtqichen/time-series-datasets

class HumanActivity(object):
    urls = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00196/ConfLongDemo_JSI.txt',
    ]

    checksums = [
        "280a677ae0d673ec9a50ada480d8a529"
    ]

    tag_ids = [
        "010-000-024-033", #"ANKLE_LEFT",
        "010-000-030-096", #"ANKLE_RIGHT",
        "020-000-033-111", #"CHEST",
        "020-000-032-221" #"BELT"
    ]
    
    tag_dict = {k: i for i, k in enumerate(tag_ids)}

    label_names = [
         "walking",
         "falling",
         "lying down",
         "lying",
         "sitting down",
         "sitting",
         "standing up from lying",
         "on all fours",
         "sitting on the ground",
         "standing up from sitting",
         "standing up from sit on grnd"
    ]

    #label_dict = {k: i for i, k in enumerate(label_names)}

    #Merge similar labels into one class
    label_dict = {
        "walking": 0,
         "falling": 1,
         "lying": 2,
         "lying down": 2,
         "sitting": 3,
         "sitting down" : 3,
         "standing up from lying": 4,
         "standing up from sitting": 4,
         "standing up from sit on grnd": 4,
         "on all fours": 5,
         "sitting on the ground": 6
         }


    def __init__(
        self, 
        root, 
        download=True,
        reduce='average', 
        max_seq_length = None,
        n_samples = None, 
        device = torch.device("cpu")
    ):

        self.root = root
        self.reduce = reduce
        # self.max_seq_length = max_seq_length

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        
        if device == torch.device("cpu"):
            self.data = torch.load(os.path.join(self.processed_folder, self.data_file), map_location='cpu')
        else:
            self.data = torch.load(os.path.join(self.processed_folder, self.data_file))

        if n_samples is not None:
            self.data = self.data[:n_samples]

    def download(self):
        from torchvision.datasets.utils import download_url
        if self._check_exists():
            return

        self.device = torch.device("cpu")

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        def save_record(records, record_id, tt, vals, mask, labels):
            tt = torch.tensor(tt).to(self.device)

            vals = torch.stack(vals)
            mask = torch.stack(mask)
            labels = torch.stack(labels)

            # flatten the measurements for different tags
            vals = vals.reshape(vals.size(0), -1)
            mask = mask.reshape(mask.size(0), -1)
            assert(len(tt) == vals.size(0))
            assert(mask.size(0) == vals.size(0))
            assert(labels.size(0) == vals.size(0))

            records.append((record_id, tt, vals, mask))


        for url, checksum in zip(self.urls, self.checksums):
            filename = url.rpartition('/')[2]
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(f"{self.raw_folder}/{filename}", 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            assert self._calculate_md5(f"{self.raw_folder}/{filename}") == checksum, f"MD5 hash check failed. Downloaded raw dataset file at {self.raw_folder}/{filename} may be broken!"

            print('Processing {}...'.format(filename))

            dirname = os.path.join(self.raw_folder)
            records = []
            first_tp = None

            for txtfile in os.listdir(dirname):
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = -1
                    tt = []

                    record_id = None
                    for l in lines:
                        cur_record_id, tag_id, time, date, val1, val2, val3, label = l.strip().split(',')
                        value_vec = torch.Tensor((float(val1), float(val2), float(val3))).to(self.device)
                        time = float(time)

                        if cur_record_id != record_id:
                            if record_id is not None:
                                save_record(records, record_id, tt, vals, mask, labels)
                            tt, vals, mask, nobs, labels = [], [], [], [], []
                            record_id = cur_record_id
                        
                            tt = [torch.zeros(1).to(self.device)]
                            vals = [torch.zeros(len(self.tag_ids),3).to(self.device)]
                            mask = [torch.zeros(len(self.tag_ids),3).to(self.device)]
                            nobs = [torch.zeros(len(self.tag_ids)).to(self.device)]
                            labels = [torch.zeros(len(self.label_names)).to(self.device)]
                            
                            first_tp = time
                            time = round((time - first_tp)/ 10**4)
                            prev_time = time
                        else:
                            time = round((time - first_tp)/ 10**4) # quatizing by 1000 ms. 10,000 is one millisecond, 10,000,000 is one second

                        if time != prev_time:
                            tt.append(time)
                            vals.append(torch.zeros(len(self.tag_ids),3).to(self.device))
                            mask.append(torch.zeros(len(self.tag_ids),3).to(self.device))
                            nobs.append(torch.zeros(len(self.tag_ids)).to(self.device))
                            labels.append(torch.zeros(len(self.label_names)).to(self.device))
                            prev_time = time

                        if tag_id in self.tag_ids:
                            n_observations = nobs[-1][self.tag_dict[tag_id]]
                            if (self.reduce == 'average') and (n_observations > 0):
                                prev_val = vals[-1][self.tag_dict[tag_id]]
                                new_val = (prev_val * n_observations + value_vec) / (n_observations + 1)
                                vals[-1][self.tag_dict[tag_id]] = new_val
                            else:
                                vals[-1][self.tag_dict[tag_id]] = value_vec

                            mask[-1][self.tag_dict[tag_id]] = 1
                            nobs[-1][self.tag_dict[tag_id]] += 1

                            if label in self.label_names:
                                if torch.sum(labels[-1][self.label_dict[label]]) == 0:
                                    labels[-1][self.label_dict[label]] = 1
                        else:
                            assert tag_id == 'RecordID', 'Read unexpected tag id {}'.format(tag_id)
                    save_record(records, record_id, tt, vals, mask, labels)
            
            print('# of records after processed:', len(records))
            torch.save(
                records,
                os.path.join(self.processed_folder, 'data.pt')
            )
                
        print('Done!')

    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition('/')[2]
            if not os.path.exists(
                os.path.join(self.processed_folder, 'data.pt')
            ):
                return False
        return True

    def _calculate_md5(
        self,
        file_path: str
    ):
        '''
        Calculate MD5 for downloaded raw dataset files
        '''
        md5_hash = hashlib.md5()
        
        with open(file_path, 'rb') as f:
            # Read the file in chunks to avoid using too much memory
            for byte_block in iter(lambda: f.read(4096), b""):
                md5_hash.update(byte_block)
        
        return md5_hash.hexdigest()
    
    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def data_file(self):
        return 'data.pt'

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        fmt_str += '    Max length: {}\n'.format(self.max_seq_length)
        fmt_str += '    Reduce: {}\n'.format(self.reduce)
        return fmt_str

def get_person_id(record_id):
    # The first letter is the person id
    person_id = record_id[0]
    person_id = ord(person_id) - ord("A")
    return person_id

def Activity_time_chunk(data, configs):

    chunk_data = []
    history = configs.seq_len # ms
    pred_window = configs.pred_len # ms
    sample_ID = 0
    for record_id, tt, vals, mask in data:
        t_max = int(tt.max())
        for st in range(0, t_max - history, 4000):
            et_x = st + history
            et_y = st + history + pred_window
            if(et_x >= t_max):
                idx_x = torch.where((tt >= st) & (tt <= et_x))[0]
            else:
                idx_x = torch.where((tt >= st) & (tt < et_x))[0]
            if(et_y >= t_max):
                idx_y = torch.where((tt >= et_x) & (tt <= et_y))[0]
            else:
                idx_y = torch.where((tt >= et_x) & (tt < et_y))[0]
            new_id = f"{record_id}_{st//pred_window}"
            # chunk_data.append((new_id, tt[idx_x] - st, vals[idx], mask[idx]))
            t_start = tt[idx_x][0]
            t_end = tt[idx_y][-1] + 1 if len(tt[idx_y]) > 0 else tt[idx_x][-1] + 1
            chunk_data.append({
                "sample_ID": sample_ID,
                "x_mark": (tt[idx_x] - t_start) / (t_end - t_start),
                "y_mark": (tt[idx_y] - t_start) / (t_end - t_start),
                "x": vals[idx_x],
                "y": vals[idx_y],
                "x_mask": mask[idx_x],
                "y_mask": mask[idx_y],
            })
            sample_ID += 1

    return chunk_data


def Activity_get_seq_length(configs, records):
    
    max_input_len = 0
    max_pred_len = 0
    lens = []
    for b, (record_id, tt, vals, mask) in enumerate(records):
        n_observed_tp = torch.lt(tt, configs.history).sum()
        max_input_len = max(max_input_len, n_observed_tp)
        max_pred_len = max(max_pred_len, len(tt) - n_observed_tp)
        lens.append(n_observed_tp)
    lens = torch.stack(lens, dim=0)
    median_len = lens.median()

    return max_input_len, max_pred_len, median_len

def variable_time_collate_fn_activity(batch, configs, device = torch.device("cpu"), data_type = "train"):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
        - record_id is a patient id
        - tt is a 1-dimensional tensor containing T time values of observations.
        - vals is a (T, D) tensor containing observed values for D variables.
        - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
        - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
        combined_tt: The union of all time observations.
        combined_vals: (M, T, D) tensor containing the observed values.
        combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """
    import lib.utils as utils
    D = batch[0][2].shape[1]
    N = batch[0][-1].shape[1] # number of labels

    combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
    combined_tt = combined_tt.to(device)

    offset = 0
    combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
    combined_labels = torch.zeros([len(batch), len(combined_tt), N]).to(device)
    
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        tt = tt.to(device)
        vals = vals.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        indices = inverse_indices[offset:offset + len(tt)]
        offset += len(tt)

        combined_vals[b, indices] = vals
        combined_mask[b, indices] = mask
        combined_labels[b, indices] = labels

    combined_tt = combined_tt.float()

    if torch.max(combined_tt) != 0.:
        combined_tt = combined_tt / torch.max(combined_tt)

    data_dict = {
        "data": combined_vals, 
        "time_steps": combined_tt,
        "mask": combined_mask,
        "labels": combined_labels}

    data_dict = utils.split_and_subsample_batch(data_dict, configs, data_type = data_type)
    return data_dict


# if __name__ == '__main__':
# 	torch.manual_seed(1991)

# 	dataset = PersonActivity('data/PersonActivity', download=True)
# 	dataloader = DataLoader(dataset, batch_size=30, shuffle=True, collate_fn= variable_time_collate_fn_activity)
# 	dataloader.__iter__().next()