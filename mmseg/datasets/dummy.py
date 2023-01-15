from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DummyDataset(CustomDataset):
    CLASSES = ('background', 'building', 'water', 'green', 'highway', 'beach', 'road')

    PALETTE = [[255, 255, 255], [128, 128, 128], [0, 0, 128], [128, 0, 0], [255, 110, 0],
               [255, 216, 0], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(DummyDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
