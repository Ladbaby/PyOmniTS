from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torchaudio.prototype.pipelines import VGGISH
from einops import rearrange
from tqdm import tqdm

from torch.utils.data import Dataset
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs
from utils.configs import configs

class Data(Dataset):
    '''
    OpenMIC dataset
    - raw dataset: https://zenodo.org/records/1432913#.W6dPeJNKjOR
    '''
    def __init__(
        self, 
        configs: ExpConfigs,
        flag: str = 'train', 
        **kwargs
    ):
        self.configs = configs
        assert flag in ['train', 'test', 'val', 'test_all']
        self.flag = flag

        self.bundle = VGGISH
        self.xs: Tensor = None
        self.x_reprs: Tensor = None
        self.y_classes: Tensor = None

        self.preprocess()

        self.dummy_x = torch.ones((self.configs.seq_len, 1))

    def __getitem__(self, index):
        return {
            # "x": self.xs[index],
            "x": self.dummy_x, # DEBUG: temporal change
            "x_repr": self.x_reprs[index],
            "y_class": self.y_classes[index]
        }

    def __len__(self):
        return len(self.y_classes)

    def preprocess(self):
        '''
        Load all audio files
        '''
        raw_audio_path = Path(self.configs.dataset_root_path) / "audio"

        logger.debug(f"Loading audio files from {raw_audio_path}")

        # load x
        all_files = []
        for folder in raw_audio_path.iterdir():
            if folder.is_dir():
                for file in folder.iterdir():
                    if file.is_file():
                        all_files.append(file)

        # Calculate the number of files for each split
        total_files = len(all_files)
        subset_ratio = 1 # DEBUG: select only a subset for training in debug
        total_files = int(total_files * subset_ratio)
        boundary_dict = {
            'train': (0, 0.9 * 0.9),
            'val': (0.9 * 0.9, 0.9),
            'test': (0.9, 1),
            'test_all': (0, 1),
        }
        left_boundary = int(total_files * boundary_dict[self.flag][0])
        right_boundary = int(total_files * boundary_dict[self.flag][1])

        # x_temp = []
        # min_length = 160000 # crop to min_length to align seq_len of different samples
        # # Iterate and load only the required subset
        # for i, file in tqdm(enumerate(all_files), total=total_files, desc="Loading"):
        #     if i < left_boundary:
        #         continue
        #     elif i >= right_boundary:
        #         break
        #     else:
        #         temp = self.load_and_preprocess_audio(file)
        #         if temp.shape[1] < min_length:
        #             min_length = temp.shape[1]
        #         x_temp.append(temp[:, :min_length])

        # self.xs = rearrange(torch.stack(x_temp), "N_SAMPLE ENC_IN SEQ_LEN -> N_SAMPLE SEQ_LEN ENC_IN")

        # load y_class
        npz_data = np.load(Path(self.configs.dataset_root_path) / self.configs.dataset_file_name)
        self.y_classes = torch.argmax(torch.from_numpy(npz_data['Y_true'][left_boundary: right_boundary]), dim=1).type(torch.LongTensor)
        self.x_reprs = torch.from_numpy(npz_data['X'][left_boundary: right_boundary] / 255.0).float()


    def load_and_preprocess_audio(self, audio_path: Path):
        """
        Load audio file and preprocess for VGGish.
        
        Args:
            audio_path: Path to MP3 file
            
        Returns:
            torch.Tensor: Preprocessed audio waveform of shape [enc_in, seq_len] (should be divided by sampling rate to obtain the length in seconds)
        """
        try:
            # Load audio using torchaudio
            waveform, orig_sr = torchaudio.load(audio_path)
        except Exception as e:
            try:
                import librosa
                # Fallback to librosa if torchaudio fails with MP3
                waveform, orig_sr = librosa.load(audio_path, sr=None)
                waveform = torch.from_numpy(waveform).unsqueeze(0)
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio file {audio_path}. "
                                 f"Torchaudio error: {e}, Librosa error: {e2}")
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to VGGish expected sample rate (16kHz)
        if orig_sr != self.bundle.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr, 
                new_freq=self.bundle.sample_rate
            )
            waveform = resampler(waveform)
        
        return waveform