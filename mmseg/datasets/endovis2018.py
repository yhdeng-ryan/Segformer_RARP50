from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp
import mmcv

@DATASETS.register_module()
class Endovis2018Dataset(CustomDataset):
    """Endovis2018Dataset dataset.

    In segmentation map annotation for RARP50, 0 stands for background, which
    is not included in 11 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'anatomy', 'instrument_shaft', 'instrument_clasper', 'instrument_wrist', 'kidney_parenchyma', 'covered_kidney', 'thread', 'clamps', 'suturing_needle', 'suction_instrument', 'small_intestine', 'ultrasound_probe')

    PALETTE = [[0,0,0], [0,255,0], [ 0,255,255], [125,255,12], [ 255,55,0],
               [24,55,125], [187,155,25], [0,255,125], [255,255,125], [ 123,15,175], [ 124,155,5], [ 12,255,141]]

    def __init__(self, **kwargs):
        super(Endovis2018Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def get_gt_seg_maps(self, efficient_test=True):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            if efficient_test:
                gt_seg_map = seg_map
            else:
                gt_seg_map = mmcv.imread(
                    seg_map, flag='unchanged', backend='pillow')
                gt_seg_map = gt_seg_map[:,:]
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps
