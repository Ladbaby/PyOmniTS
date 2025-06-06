import torch
import torch.nn as nn
from torch import Tensor
from torchaudio.prototype.pipelines import VGGISH
from einops import rearrange

from utils.ExpConfigs import ExpConfigs
from utils.globals import logger

class Model(nn.Module):
    """
    warpper for pretrained pipeline torchaudio.prototype.pipelines.VGGISH
    - paper: "CNN architectures for large-scale audio classification" (ICASSP 2017)
    - paper link: http://ieeexplore.ieee.org/document/7952132/
    """
    def __init__(self, configs: ExpConfigs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.configs = configs

        assert self.task_name in ["representation_learning"], "VGGish only suppports '--task_name representation_learning'"
        assert self.configs.is_training == 0, "VGGish only suppports '--is_training 0'. Do not train it!"

        self.bundle = VGGISH
        try:
            self.model = self.bundle.get_model()
        except Exception as e:
            logger.exception(e)
            logger.warning(f"It is possible that downloaded weight for VGGish under `~/.cache/torch/hub/torchaudio/models/` is broken. Try manually remove it then rerun.")
            exit(1)
        self.input_proc = VGGISH.get_input_processor()

        self.post_proc = Postprocessor()

    def forward(
        self, 
        x: Tensor,
        **kwargs
    ):
        self.model.eval()

        # Convert to mono if stereo
        x = torch.mean(x, dim=-1) # (BATCH_SIZE, SEQ_LEN)

        output = []
        for sample in x:
            # The VGGish model expects input of shape [time_length]
            output.append(self.post_proc(self.model(self.input_proc(sample))) / 255.0)
        
        if self.configs.task_name in ["representation_learning"]:
            return {
                "pred_repr_time": torch.stack(output),
            }
        else:
            raise NotImplementedError

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
            (128, 128,),
            dtype=torch.float,
        )
        self.pca_means = torch.empty(
            (128, 1), dtype=torch.float
        )

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)

        self.load_state_dict(torch.load("storage/pretrained/OpenMIC/VGGish/vggish_pca_params.pth"))


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
            embeddings_batch.shape[1] == 128
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
            pca_applied, -2.0, +2.0
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round(
            (clipped_embeddings - -2.0)
            * (
                255.0
                / (+2.0 - -2.0)
            )
        )
        return torch.squeeze(quantized_embeddings)

    def forward(self, x):
        return self.postprocess(x)