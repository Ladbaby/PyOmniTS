import os
from pathlib import Path
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from torch import hub
from torchvision.datasets.utils import download_url

from . import vggish_input, vggish_params
from utils.globals import logger

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True))

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)


class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty(
            (vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE,),
            dtype=torch.float,
        )
        self.pca_means = torch.empty(
            (vggish_params.EMBEDDING_SIZE, 1), dtype=torch.float
        )

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)

        current_directory = os.getcwd()
        folder_name = os.path.basename(current_directory)
        if folder_name == "backend":
            # cwd in backend case
            base_path = ""
        else:
            # cwd in InsRec root case
            base_path = "backend/"
        checkpoint_file = Path(f"{base_path}storage/pretrained/OpenMIC/VGGish/vggish_pca_params.pth")
        if not checkpoint_file.exists():
            # try downloading from web 
            url = "https://huggingface.co/Ladbaby/InsRec-models/resolve/main/OpenMIC/VGGish/vggish_pca_params.pth?download=true"
            download_choice = input(f'''The checkpoint file for the Postprocesor of VGGish is going to be downloaded at "{checkpoint_file}" via url {url}, proceed? (Y/N):''')
            while True:
                if download_choice.upper() == 'Y':
                    break
                elif download_choice.upper() == 'N':
                    logger.info("Download aborted.")
                    exit(0)
                else:
                    download_choice = input(f"Invalid choice '{download_choice}', please select between Y and N:")

            # Read proxy settings from environment variables (both lowercase and uppercase)
            http_proxy = os.environ.get('http_proxy') or os.environ.get('HTTP_PROXY')
            https_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')

            proxies = {}
            if http_proxy:
                proxies['http'] = http_proxy
                proxies['https'] = http_proxy # if https_proxy is not present, use http_proxy for https traffic instead
            if https_proxy:
                proxies['https'] = https_proxy

            # Install proxy handler if proxies are found
            if proxies:
                logger.debug(f"Using proxy read from environments for download: {proxies}")
                proxy_handler = urllib.request.ProxyHandler(proxies)
                opener = urllib.request.build_opener(proxy_handler)
                urllib.request.install_opener(opener)
            try:
                download_url(
                    url=url,
                    root=f"{base_path}storage/pretrained/OpenMIC/VGGish",
                    filename="vggish_pca_params.pth"
                )
            except Exception as e:
                logger.exception(e, stack_info=True)
        try:
            self.load_state_dict(torch.load(checkpoint_file))
        except Exception as e:
            logger.exception(e, stack_info=True)
            logger.exception(f"Postprocessor of VGGish failed to load checkpoint file at '{checkpoint_file}'.")
            logger.warning(f"It is possible that the checkpoint file at '{checkpoint_file}' is broken. Try manually remove it then rerun.")

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
            embeddings_batch.shape,
        )
        assert (
            embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE
        ), "Bad batch shape: %r" % (embeddings_batch.shape,)

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t()

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(
            pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round(
            (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)
            * (
                255.0
                / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)
            )
        )
        return torch.squeeze(quantized_embeddings)

    def forward(self, x):
        return self.postprocess(x)


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg():
    return VGG(make_layers())


# def _spectrogram():
#     config = dict(
#         sr=16000,
#         n_fft=400,
#         n_mels=64,
#         hop_length=160,
#         window="hann",
#         center=False,
#         pad_mode="reflect",
#         htk=True,
#         fmin=125,
#         fmax=7500,
#         output_format='Magnitude',
#         #             device=device,
#     )
#     return Spectrogram.MelSpectrogram(**config)


class VGGish(VGG):
    def __init__(self, urls, pretrained=True, preprocess=True, postprocess=True, progress=True):
        super().__init__(make_layers())
        if pretrained:
            state_dict = hub.load_state_dict_from_url(urls['vggish'], progress=progress)
            super().load_state_dict(state_dict)

        self.preprocess = preprocess
        self.postprocess = postprocess
        if self.postprocess:
            self.pproc = Postprocessor()
            if pretrained:
                state_dict = hub.load_state_dict_from_url(urls['pca'], progress=progress)
                # TODO: Convert the state_dict to torch
                state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME] = torch.as_tensor(
                    state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME], dtype=torch.float
                )
                state_dict[vggish_params.PCA_MEANS_NAME] = torch.as_tensor(
                    state_dict[vggish_params.PCA_MEANS_NAME].reshape(-1, 1), dtype=torch.float
                )

                self.pproc.load_state_dict(state_dict)

    def forward(self, x, fs=None):
        if self.preprocess:
            x = self._preprocess(x, fs)
        x = VGG.forward(self, x)
        if self.postprocess:
            x = self._postprocess(x)
        return x

    def _preprocess(self, x, fs):
        if isinstance(x, np.ndarray):
            x = vggish_input.waveform_to_examples(x, fs)
        elif isinstance(x, str):
            x = vggish_input.wavfile_to_examples(x)
        else:
            raise AttributeError
        return x

    def _postprocess(self, x):
        return self.pproc(x)
