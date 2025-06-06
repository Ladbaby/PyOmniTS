dependencies = ['torch', 'numpy', 'resampy', 'soundfile']

from layers.VGGish.torchvggish.vggish import VGGish

model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}


def vggish(**kwargs):
    model = VGGish(urls=model_urls, **kwargs)
    return model
