from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ImaginaireDataset(CustomDataset):
    CLASSES = ('background', 'building', 'road', 'green', 'water', 'beach')

    PALETTE = [[255, 255, 255], [128, 128, 128], [128, 0, 0], [0, 128, 0], [0, 0, 128],
               [255, 216, 0]]

    def __init__(self, **kwargs):
        super(ImaginaireDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
