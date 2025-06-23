import numpy as np
import torch
import mmcv
import os.path as osp

from mmseg.datasets import PIPELINES
from mmseg.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC


@PIPELINES.register_module()
class ImageToTensor_clips(object):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """

        for key in self.keys:
            assert isinstance(results[key], list)
            img_all=[]
            for im_one in results[key]:
                img = im_one
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img=to_tensor(img.transpose(2, 0, 1))
                img_all.append(img)
            results[key] = img_all
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class DefaultFormatBundle_clips(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            assert isinstance(results['img'], list)
            img_all=[]
            for im in results['img']:
                # img = results['img']
                if len(im.shape) < 3:
                    im = np.expand_dims(im, -1)
                img = np.ascontiguousarray(im.transpose(2, 0, 1))
                img_all.append(to_tensor(img))
            img_all=torch.stack(img_all)
            results['img'] = DC(img_all, stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            gt_seg_all=[]
            assert isinstance(results['gt_semantic_seg'], list)
            for gt in results['gt_semantic_seg']:
                gt_one= to_tensor(gt[None,
                                                         ...].astype(np.int64))
                    
                gt_seg_all.append(gt_one)
            gt_seg_all=torch.stack(gt_seg_all)
            results['gt_semantic_seg']=DC(gt_seg_all, stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__

@PIPELINES.register_module()
class DefaultFormatBundle_clips2(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        if 'img' in results:
            assert isinstance(results['img'], list)
            img_all=[]
            for im in results['img']:
                # img = results['img']
                if len(im.shape) < 3:
                    im = np.expand_dims(im, -1)
                img = np.ascontiguousarray(im.transpose(2, 0, 1))
                img_all.append(to_tensor(img))
            img_all=torch.stack(img_all)

            assert isinstance(results['img_beforeDistortion'], list)
            img_all_old=[]
            for im in results['img_beforeDistortion']:
                # img = results['img']
                if len(im.shape) < 3:
                    im = np.expand_dims(im, -1)
                img = np.ascontiguousarray(im.transpose(2, 0, 1))
                img_all_old.append(to_tensor(img))
            img_all_old=torch.stack(img_all_old)

            img_all=torch.stack([img_all,img_all_old])
            results['img'] = DC(img_all, stack=True)
        if 'gt_semantic_seg' in results:
            # convert to long
            gt_seg_all=[]
            assert isinstance(results['gt_semantic_seg'], list)
            for gt in results['gt_semantic_seg']:
                gt_one= to_tensor(gt[None,
                                                         ...].astype(np.int64))
                    
                gt_seg_all.append(gt_one)
            gt_seg_all=torch.stack(gt_seg_all)
            results['gt_semantic_seg']=DC(gt_seg_all, stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__



@PIPELINES.register_module()
class LoadAnnotations_ivps(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='grayscale',
            backend=self.imdecode_backend).squeeze().astype(np.float32)
        gt_semantic_seg = gt_semantic_seg/255.0
        # print(gt_semantic_seg.shape)
        assert gt_semantic_seg.ndim==2, '%s'%(filename)
        assert gt_semantic_seg.shape[0]==results['img'].shape[0] and gt_semantic_seg.shape[1]==results['img'].shape[1]
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
