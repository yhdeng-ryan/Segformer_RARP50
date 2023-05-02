from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp
import mmcv

@DATASETS.register_module()
class RARP50Dataset(CustomDataset):
    """RARP50 dataset.

    In segmentation map annotation for RARP50, 0 stands for background, which
    is not included in 9 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = (
        'tool_clasper', 'tool_wrist', 'tool_shaft', 'suturing_needle', 'thread', 'suction_tool', 'needle_holder', 'clamps', 'catheter')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230]]

    def __init__(self, **kwargs):
        super(RARP50Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
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
                gt_seg_map = gt_seg_map[:,:,0]
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps