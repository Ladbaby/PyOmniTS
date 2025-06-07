import math
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torchaudio.prototype.pipelines import VGGISH
from einops import repeat, rearrange

from torch.utils.data import Dataset
from utils.globals import logger
from utils.ExpConfigs import ExpConfigs

class Data(Dataset):
    '''
    OpenMIC dataset
    - raw dataset: https://zenodo.org/records/1432913#.W6dPeJNKjOR

    classes:
    0. "🪗 Accordion"
    1. "🪕 Banjo"
    2. "Bass"
    3. "Cello"
    4. "Clarinet"
    5. "Cymbals"
    6. "🥁 Drums"
    7. "Flute"
    8. "🎸 Guitar"
    9. "Mallet Percussion"
    10. "Mandolin"
    11. "Organ"
    12. "🎹 Piano"
    13. "🎷 Saxophone"
    14. "Synthesizer"
    15. "Trombone"
    16. "🎺 Trumpet"
    17. "Ukulele"
    18. "🎻 Violin"
    19. "🗣️ Voice"
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
        self.x_masks: Tensor = None
        self.x_repr_times: Tensor = None
        self.y_classes: Tensor = None

        self.load_xs = True # load downsampled audio files
        self.x_dummy = torch.ones(self.configs.sampling_rate * 10, 1)

        self.preprocess()

    def __getitem__(self, index):
        temp_dict = {}
        if self.xs is not None:
            temp_dict["x"] = self.xs[index]
        else:
            temp_dict["x"] = self.x_dummy

        if self.x_masks is not None:
            temp_dict["x_mask"] = self.x_masks[index]
        else:
            temp_dict["x_mask"] = self.x_dummy

        if self.x_repr_times is not None:
            temp_dict["x_repr_time"] = self.x_repr_times[index]

        if self.y_classes is not None:
            temp_dict["y_class"] = self.y_classes[index]

        return temp_dict

    def __len__(self):
        return len(self.y_classes) if self.y_classes is not None else len(self.xs)

    def preprocess(self):
        '''
        Load all audio files
        '''
        processed_audio_path = Path(self.configs.dataset_root_path) / "processed"

        total_files = 20000
        boundary_dict = {
            'train': (0, 0.9 * 0.9),
            'val': (0.9 * 0.9, 0.9),
            'test': (0.9, 1),
            'test_all': (0, 1),
        }
        left_boundary = int(total_files * boundary_dict[self.flag][0])
        right_boundary = int(total_files * boundary_dict[self.flag][1])

        # load y_class
        try:
            npz_data = np.load(Path(self.configs.dataset_root_path) / self.configs.dataset_file_name, allow_pickle=True)


            # [N_SAMPLE, N_CLASSES]
            self.y_classes = torch.from_numpy(npz_data['Y_true'][left_boundary: right_boundary] * npz_data['Y_mask'][left_boundary: right_boundary])
            self.y_classes = self.y_classes.float()

            # self.x_repr_times = torch.from_numpy(npz_data['X'][left_boundary: right_boundary] / 255.0).float()
            self.x_repr_times = torch.from_numpy(np.load(processed_audio_path / "x_repr_times.npy")[left_boundary: right_boundary] / 255.0).float()

            # DEBUG
            # self.sample_keys = npz_data['sample_key'][left_boundary: right_boundary]
            # for class_tensor, sample_key, Y_true, Y_mask in zip(self.y_classes, self.sample_keys, npz_data['Y_true'][left_boundary: right_boundary], npz_data['Y_mask'][left_boundary: right_boundary]):
            #     logger.debug(f"{class_tensor=}")
            #     logger.debug(f"{sample_key=}")
            #     logger.debug(f"{Y_true=}")
            #     logger.debug(f"{Y_mask=}")
            #     input()
        except Exception as e:
            logger.warning(f"{e}", stack_info=True)
            logger.warning(f"You can ignore the above warning, if you are only running inference instead of train/val/test")

        
        if self.load_xs and processed_audio_path.exists():
            try:
                logger.debug(f"Loading audio files from {processed_audio_path / 'xs.npy'}")
                xs_temp = np.load(processed_audio_path / "xs.npy")[left_boundary:right_boundary]

                self.xs = repeat(torch.from_numpy(xs_temp), "N_SAMPLE SEQ_LEN -> N_SAMPLE SEQ_LEN ENC_IN", ENC_IN=1)
            except Exception as e:
                logger.warning(f"{e}", stack_info=True)
                logger.warning(f"You can ignore the above warning, if you are only running inference instead of train/val/test")


    def load_and_preprocess_audio(self, audio: BinaryIO | Path):
        """
        Load audio file and preprocess for VGGish.
        
        Args:
            audio_path: Path to MP3 file
            
        Returns:
            torch.Tensor: Preprocessed audio waveform of shape [enc_in, seq_len] (should be divided by sampling rate to obtain the length in seconds)
        """
        try:
            # Load audio using torchaudio
            waveform, orig_sr = torchaudio.load(audio)
        except Exception as e:
            try:
                import librosa
                # Fallback to librosa if torchaudio fails with MP3
                waveform, orig_sr = librosa.load(audio, sr=None)
                waveform = torch.from_numpy(waveform).unsqueeze(0)
            except Exception as e2:
                raise RuntimeError(f"Failed to load audio file {audio}. "
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

    def load_custom_data(self, audio: BinaryIO | Path):
        self.inference_flag = True
        waveform = self.load_and_preprocess_audio(audio) # [enc_in, seq_len]
        waveform = rearrange(waveform, "ENC_IN SEQ_LEN -> SEQ_LEN ENC_IN")

        SEQ_LEN = self.configs.sampling_rate * 10 # 16k sampling rate * 10 seconds

        def split_tensor(tensor, seq_len):
            time_length, enc_in = tensor.shape
            n_samples = math.ceil(time_length / seq_len)

            # Create lists to hold the samples and masks
            samples = []
            masks = []

            for i in range(n_samples):
                start_idx = i * seq_len
                end_idx = start_idx + seq_len
                
                # Slice the tensor
                sample = tensor[start_idx:end_idx]
                
                # Create mask for this sample (1 for real data, 0 for padding)
                mask = torch.ones(sample.shape[0])
                
                # Pad if the sample is shorter than seq_len
                if sample.shape[0] < seq_len:
                    padding_length = seq_len - sample.shape[0]
                    padding = torch.zeros(padding_length, enc_in)
                    sample = torch.cat((sample, padding), dim=0)
                    
                    # Add padding mask (0 for padded positions)
                    padding_mask = torch.zeros(seq_len - mask.shape[0])
                    mask = torch.cat((mask, padding_mask), dim=0)
                
                samples.append(sample)
                masks.append(mask)

            # Stack all samples and masks into single tensors
            output_tensor = torch.stack(samples)
            output_mask = torch.stack(masks)

            # Ensure the first dimension is divisible by BATCH_SIZE
            n_samples_final = output_tensor.shape[0]
            if n_samples_final % self.configs.batch_size != 0:
                # Calculate how many more samples are needed
                needed_samples = self.configs.batch_size - (n_samples_final % self.configs.batch_size)
                
                # Replicate last sample and create corresponding mask (all zeros for batch padding)
                last_sample = output_tensor[-1].unsqueeze(0)
                additional_samples = last_sample.repeat(needed_samples, 1, 1)
                output_tensor = torch.cat((output_tensor, additional_samples), dim=0)
                
                # Create mask for batch padding (all zeros since these are entirely padded samples)
                batch_padding_mask = torch.zeros(needed_samples, output_tensor.shape[1])
                output_mask = torch.cat((output_mask, batch_padding_mask), dim=0)
            
            return output_tensor, output_mask.unsqueeze(-1)

        self.xs, self.x_masks = split_tensor(tensor=waveform, seq_len=SEQ_LEN)
