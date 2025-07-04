# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import numpy as np
import torch
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample, ReIDDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes



@TRANSFORMS.register_module()
class PackDetInputs(BaseTransform):
    """Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        #print(f"Received results in Packdetinputs: {results}")
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = img

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class ToTensor:
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert data in results to :obj:`torch.Tensor`.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data converted
                to :obj:`torch.Tensor`.
        """
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class ImageToTensor:
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
                to :obj:`torch.Tensor` and permuted to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img).permute(2, 0, 1).contiguous()

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@TRANSFORMS.register_module()
class Transpose:
    """Transpose some results by given keys.

    Args:
        keys (Sequence[str]): Keys of results to be transposed.
        order (Sequence[int]): Order of transpose.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def __call__(self, results):
        """Call function to transpose the channel order of data in results.

        Args:
            results (dict): Result dict contains the data to transpose.

        Returns:
            dict: The result dict contains the data transposed to \
                ``self.order``.
        """
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, order={self.order})'


@TRANSFORMS.register_module()
class WrapFieldsToLists:
    """Wrap fields of the data dictionary into lists for evaluation.

    This class can be used as a last step of a test or validation
    pipeline for single image evaluation or inference.

    Example:
        >>> test_pipeline = [
        >>>    dict(type='LoadImageFromFile'),
        >>>    dict(type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
        >>>    dict(type='Pad', size_divisor=32),
        >>>    dict(type='ImageToTensor', keys=['img']),
        >>>    dict(type='Collect', keys=['img']),
        >>>    dict(type='WrapFieldsToLists')
        >>> ]
    """

    def __call__(self, results):
        """Call function to wrap fields into lists.

        Args:
            results (dict): Result dict contains the data to wrap.

        Returns:
            dict: The result dict where value of ``self.keys`` are wrapped \
                into list.
        """

        # Wrap dict fields into lists
        for key, val in results.items():
            results[key] = [val]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}()'


@TRANSFORMS.register_module()
class PackTrackInputs(BaseTransform):
    """Pack the inputs data for the multi object tracking and video instance
    segmentation. All the information of images are packed to ``inputs``. All
    the information except images are packed to ``data_samples``. In order to
    get the original annotaiton and meta info, we add `instances` key into meta
    keys.

    Args:
        meta_keys (Sequence[str]): Meta keys to be collected in
            ``data_sample.metainfo``. Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('img_id',
            'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'frame_id', 'is_video_data',
            'video_id', 'video_length', 'instances').
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_instances_ids': 'instances_ids'
    }

    def __init__(self,
                 meta_keys: Optional[dict] = None,
                 default_meta_keys: tuple = ('img_id', 'img_path', 'ori_shape',
                                             'img_shape', 'scale_factor',
                                             'flip', 'flip_direction',
                                             'frame_id', 'video_id',
                                             'video_length',
                                             'ori_video_length', 'instances')):
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.
        Args:
            results (dict): Result dict from the data pipeline.
        Returns:
            dict:
            - 'inputs' (dict[Tensor]): The forward data of models.
            - 'data_samples' (obj:`TrackDataSample`): The annotation info of
                the samples.
        """
        packed_results = dict()
        packed_results['inputs'] = dict()

        # 1. Pack images
        if 'img' in results:
            imgs = results['img']
            imgs = np.stack(imgs, axis=0)
            imgs = imgs.transpose(0, 3, 1, 2)
            packed_results['inputs'] = to_tensor(imgs)

        # 2. Pack InstanceData
        if 'gt_ignore_flags' in results:
            gt_ignore_flags_list = results['gt_ignore_flags']
            valid_idx_list, ignore_idx_list = [], []
            for gt_ignore_flags in gt_ignore_flags_list:
                valid_idx = np.where(gt_ignore_flags == 0)[0]
                ignore_idx = np.where(gt_ignore_flags == 1)[0]
                valid_idx_list.append(valid_idx)
                ignore_idx_list.append(ignore_idx)

        assert 'img_id' in results, "'img_id' must contained in the results "
        'for counting the number of images'

        num_imgs = len(results['img_id'])
        instance_data_list = [InstanceData() for _ in range(num_imgs)]
        ignore_instance_data_list = [InstanceData() for _ in range(num_imgs)]

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks':
                mapped_key = self.mapping_table[key]
                gt_masks_list = results[key]
                if 'gt_ignore_flags' in results:
                    for i, gt_mask in enumerate(gt_masks_list):
                        valid_idx, ignore_idx = valid_idx_list[
                            i], ignore_idx_list[i]
                        instance_data_list[i][mapped_key] = gt_mask[valid_idx]
                        ignore_instance_data_list[i][mapped_key] = gt_mask[
                            ignore_idx]

                else:
                    for i, gt_mask in enumerate(gt_masks_list):
                        instance_data_list[i][mapped_key] = gt_mask

            else:
                anns_list = results[key]
                if 'gt_ignore_flags' in results:
                    for i, ann in enumerate(anns_list):
                        valid_idx, ignore_idx = valid_idx_list[
                            i], ignore_idx_list[i]
                        instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(
                                ann[valid_idx])
                        ignore_instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(
                                ann[ignore_idx])
                else:
                    for i, ann in enumerate(anns_list):
                        instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(ann)

        det_data_samples_list = []
        for i in range(num_imgs):
            det_data_sample = DetDataSample()
            det_data_sample.gt_instances = instance_data_list[i]
            det_data_sample.ignored_instances = ignore_instance_data_list[i]
            det_data_samples_list.append(det_data_sample)

        # 3. Pack metainfo
        for key in self.meta_keys:
            if key not in results:
                continue
            img_metas_list = results[key]
            for i, img_meta in enumerate(img_metas_list):
                det_data_samples_list[i].set_metainfo({f'{key}': img_meta})

        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = det_data_samples_list
        if 'key_frame_flags' in results:
            key_frame_flags = np.asarray(results['key_frame_flags'])
            key_frames_inds = np.where(key_frame_flags)[0].tolist()
            ref_frames_inds = np.where(~key_frame_flags)[0].tolist()
            track_data_sample.set_metainfo(
                dict(key_frames_inds=key_frames_inds))
            track_data_sample.set_metainfo(
                dict(ref_frames_inds=ref_frames_inds))

        packed_results['data_samples'] = track_data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'meta_keys={self.meta_keys}, '
        repr_str += f'default_meta_keys={self.default_meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackReIDInputs(BaseTransform):
    """Pack the inputs data for the ReID. The ``meta_info`` item is always
    populated. The contents of the ``meta_info`` dictionary depends on
    ``meta_keys``. By default this includes:

        - ``img_path``: path to the image file.
        - ``ori_shape``: original shape of the image as a tuple (H, W).
        - ``img_shape``: shape of the image input to the network as a tuple
            (H, W). Note that images may be zero padded on the bottom/right
          if the batch tensor is larger than this shape.
        - ``scale``: scale of the image as a tuple (W, H).
        - ``scale_factor``: a float indicating the pre-processing scale.
        -  ``flip``: a boolean indicating if image flip transform was used.
        - ``flip_direction``: the flipping direction.
    Args:
        meta_keys (Sequence[str], optional): The meta keys to saved in the
            ``metainfo`` of the packed ``data_sample``.
    """
    default_meta_keys = ('img_path', 'ori_shape', 'img_shape', 'scale',
                         'scale_factor')

    def __init__(self, meta_keys: Sequence[str] = ()) -> None:
        self.meta_keys = self.default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple.'
            self.meta_keys += meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.
        Args:
            results (dict): Result dict from the data pipeline.
        Returns:
            dict:
            - 'inputs' (dict[Tensor]): The forward data of models.
            - 'data_samples' (obj:`ReIDDataSample`): The meta info of the
                sample.
        """
        packed_results = dict(inputs=dict(), data_samples=None)
        assert 'img' in results, 'Missing the key ``img``.'
        _type = type(results['img'])
        label = results['gt_label']

        if _type == list:
            img = results['img']
            label = np.stack(label, axis=0)  # (N,)
            assert all([type(v) == _type for v in results.values()]), \
                'All items in the results must have the same type.'
        else:
            img = [results['img']]

        img = np.stack(img, axis=3)  # (H, W, C, N)
        img = img.transpose(3, 2, 0, 1)  # (N, C, H, W)
        img = np.ascontiguousarray(img)

        packed_results['inputs'] = to_tensor(img)

        data_sample = ReIDDataSample()
        data_sample.set_gt_label(label)

        meta_info = dict()
        for key in self.meta_keys:
            meta_info[key] = results[key]
        data_sample.set_metainfo(meta_info)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str

@TRANSFORMS.register_module()
class PackTemporalInputs(BaseTransform):
    """Pack temporal inputs for object detection tasks.
    
    Properly handles both video sequences and single frames from BaseVideoDataset,
    processing all frames' annotations and maintaining temporal consistency.
    """

    def __init__(self,
                 meta_keys: Sequence[str] = ('img_id', 'video_id', 'id',
                                           'ori_shape', 'img_shape',
                                           'pad_shape', 'scale_factor',
                                           'flip', 'flip_direction', 'file_name', 'frame_number', 'sequence_name'),
                 default_meta_keys: Sequence[str] = ('filename', 'ori_filename',
                                                   'img_norm_cfg',
                                                   'is_video_data'),
                 keys: Sequence[str] = ('img', 'gt_bboxes', 'gt_bboxes_labels',
                                      'gt_masks', 'gt_ignore_flags')):
        self.meta_keys = meta_keys
        self.default_meta_keys = default_meta_keys
        self.keys = keys
        self.mapping_table = {
            'gt_bboxes': 'bboxes',
            'gt_bboxes_labels': 'labels',
            'gt_masks': 'masks'
        }

    def transform(self, results: dict) -> dict:
        """Pack temporal detection inputs into proper structures."""
        packed_results = {}
        
        # Handle image data
        packed_results['inputs'] = results['img']

        # Create data sample structure
        data_sample = DetDataSample()
        
        # Handle instance data (bboxes, labels, masks)
        if 'instances' in results:  # Single frame case
            self._pack_frame_data(results, data_sample)
        else:  # No instances (test mode)
            data_sample.gt_instances = InstanceData()

        # Handle metadata
        meta = {}
        for key in self.meta_keys:
            if key in results:
                meta[key] = results[key]
        
        # Add additional metadata
        for key in self.default_meta_keys:
            if key in results:
                meta[key] = results[key]
        
        # Add video-specific metadata
        if 'video_id' in results:
            meta['video_id'] = results['video_id']
        if 'video_length' in results:
            meta['video_length'] = results['video_length']
        if 'sequence_name' in results:
            meta['sequence_name'] = results['sequence_name']
        
        data_sample.set_metainfo(meta)
        packed_results['data_samples'] = data_sample
        
        #print(f"PackTemporalInputs: {packed_results}")
        
        return packed_results

    def _pack_frame_data(self, frame_data: dict, data_sample: DetDataSample):
        """Pack instance data from a single frame."""
        instance_data = InstanceData()
        
        if 'instances' in frame_data and len(frame_data['instances']) > 0:
            bboxes = []
            labels = []
            ignore_flags = []
            
            for instance in frame_data['instances']:
                bboxes.append(instance['bbox'])
                labels.append(instance['bbox_label'])
                ignore_flags.append(instance.get('ignore_flag', 0))
            
            if bboxes:
                instance_data.bboxes = to_tensor(np.array(bboxes, dtype=np.float32))
                instance_data.labels = to_tensor(np.array(labels, dtype=np.int64))
                instance_data.ignore_flags = to_tensor(np.array(ignore_flags, dtype=bool))
        
        # Also check for pre-processed bboxes (from previous transforms)
        if 'gt_bboxes' in frame_data:
            instance_data.bboxes = frame_data['gt_bboxes']
            instance_data.labels = to_tensor(frame_data['gt_bboxes_labels'])
            instance_data.ignore_flags = to_tensor(frame_data['gt_ignore_flags'])
            if 'gt_masks' in frame_data:
                instance_data.masks = frame_data['gt_masks']
        
        data_sample.gt_instances = instance_data

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys}, '
        repr_str += f'default_meta_keys={self.default_meta_keys}, '
        repr_str += f'keys={self.keys})'
        return repr_str

'''
@TRANSFORMS.register_module()
class PackTemporalInputs(BaseTransform):
    """Pack temporal inputs for object detection tasks.
    
    Combines functionality of VideoCollect and SeqDefaultFormatBundle while using
    InstanceData for modern MMDetection versions.
    """

    def __init__(self,
                 meta_keys: Sequence[str] = ('img_id', 'video_id', 'frame_id',
                                           'ori_shape', 'img_shape',
                                           'pad_shape', 'scale_factor',
                                           'flip', 'flip_direction'),
                 default_meta_keys: Sequence[str] = ('filename', 'ori_filename',
                                                   'img_norm_cfg',
                                                   'is_video_data'),
                 keys: Sequence[str] = ('img', 'gt_bboxes', 'gt_bboxes_labels',
                                      'gt_masks', 'gt_ignore_flags')):
        self.meta_keys = meta_keys
        self.default_meta_keys = default_meta_keys
        self.keys = keys
        self.mapping_table = {
            'gt_bboxes': 'bboxes',
            'gt_bboxes_labels': 'labels',
            'gt_masks': 'masks'
        }

    def transform(self, results: dict) -> dict:
        """Pack temporal detection inputs into proper structures.
        
        Args:
            results (dict): Result dict from loading pipeline.
            
        Returns:
            dict: Packed results with 'inputs' and 'data_samples' keys.
        """
        packed_results = {}
        
        # Handle image data
        if 'img' in results:
            packed_results['inputs'] = results['img']

        # Create data sample structure
        data_sample = DetDataSample()
        
        # Handle instance data (bboxes, labels, masks)
        if 'images' in results:  # Video case - use first frame annotations
            frame_data = results['images'][0]
            self._pack_instance_data(frame_data, data_sample)
        else:  # Single frame case
            #print(f"Received from Pad in PackTemporalInputs: {results}")
            self._pack_instance_data(results, data_sample)

        # Handle metadata
        meta = {}
        for key in self.meta_keys + self.default_meta_keys:
            if key in results:
                meta[key] = results[key]
            elif 'images' in results and len(results['images']) > 0:
                if key in results['images'][0]:
                    meta[key] = results['images'][0][key]
        
        data_sample.set_metainfo(meta)
        packed_results['data_samples'] = data_sample
        
        print(f"Packed results: {packed_results}")
        return packed_results

    def _pack_instance_data(self, frame_data: dict, data_sample: DetDataSample):
        """Pack instance data into InstanceData structures."""
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()
        
        # Handle empty bboxes case
        if 'gt_bboxes' not in frame_data or len(frame_data['gt_bboxes']) == 0:
            data_sample.gt_instances = instance_data
            return

        # Handle valid and ignored instances
        if 'gt_ignore_flags' in frame_data:
            valid_idx = np.where(frame_data['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(frame_data['gt_ignore_flags'] == 1)[0]
        else:
            valid_idx = slice(None)  # All instances
            ignore_idx = []  # No ignored instances

        # Pack instance annotations
        for key in self.mapping_table.keys():
            if key not in frame_data:
                continue
                
            data = frame_data[key]
            target_key = self.mapping_table[key]
            
            if key == 'gt_masks' or isinstance(data, BaseBoxes):
                if 'gt_ignore_flags' in frame_data:
                    instance_data[target_key] = data[valid_idx]
                    ignore_instance_data[target_key] = data[ignore_idx]
                else:
                    instance_data[target_key] = data
            else:
                if 'gt_ignore_flags' in frame_data:
                    instance_data[target_key] = to_tensor(data[valid_idx])
                    ignore_instance_data[target_key] = to_tensor(data[ignore_idx])
                else:
                    instance_data[target_key] = to_tensor(data)

        data_sample.gt_instances = instance_data
        if len(ignore_instance_data) > 0:
            data_sample.ignored_instances = ignore_instance_data

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys}, '
        repr_str += f'default_meta_keys={self.default_meta_keys}, '
        repr_str += f'keys={self.keys})'
        return repr_str
    '''

@TRANSFORMS.register_module()
class MultiImagesToTensor(object):
    """Multi images to tensor.

    1. Transpose and convert image/multi-images to Tensor.
    2. Add prefix to every key in the second dict of the inputs. Then, add
    these keys and corresponding values into the outputs.

    Args:
        ref_prefix (str): The prefix of key added to the second dict of inputs.
            Defaults to 'ref'.
    """

    def __init__(self, ref_prefix='ref'):
        self.ref_prefix = ref_prefix

    def __call__(self, results):
        """Multi images to tensor.

        1. Transpose and convert image/multi-images to Tensor.
        2. Add prefix to every key in the second dict of the inputs. Then, add
        these keys and corresponding values into the output dict.

        Args:
            results (list[dict]): List of two dicts.

        Returns:
            dict: Each key in the first dict of `results` remains unchanged.
            Each key in the second dict of `results` adds `self.ref_prefix`
            as prefix.
        """
        outs = []
        for _results in results:
            _results = self.images_to_tensor(_results)
            outs.append(_results)

        data = {}
        data.update(outs[0])
        if len(outs) == 2:
            for k, v in outs[1].items():
                data[f'{self.ref_prefix}_{k}'] = v

        return data

    def images_to_tensor(self, results):
        """Transpose and convert images/multi-images to Tensor."""
        if 'img' in results:
            img = results['img']
            if len(img.shape) == 3:
                # (H, W, 3) to (3, H, W)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
            else:
                # (H, W, 3, N) to (N, 3, H, W)
                img = np.ascontiguousarray(img.transpose(3, 2, 0, 1))
            results['img'] = to_tensor(img)
        if 'proposals' in results:
            results['proposals'] = to_tensor(results['proposals'])
        if 'img_metas' in results:
            results['img_metas'] = DC(results['img_metas'], cpu_only=True)
        return results


@TRANSFORMS.register_module()
class SeqDefaultFormatBundle(object):
    """Sequence Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "img_metas", "proposals", "gt_bboxes", "gt_instance_ids",
    "gt_match_indices", "gt_bboxes_ignore", "gt_labels", "gt_masks",
    "gt_semantic_seg" and 'padding_mask'.
    These fields are formatted as follows.

    - img: (1) transpose, (2) to tensor, (3) to DataContainer (stack=True)
    - img_metas: (1) to DataContainer (cpu_only=True)
    - proposals: (1) to tensor, (2) to DataContainer
    - gt_bboxes: (1) to tensor, (2) to DataContainer
    - gt_instance_ids: (1) to tensor, (2) to DataContainer
    - gt_match_indices: (1) to tensor, (2) to DataContainer
    - gt_bboxes_ignore: (1) to tensor, (2) to DataContainer
    - gt_labels: (1) to tensor, (2) to DataContainer
    - gt_masks: (1) to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1) unsqueeze dim-0 (2) to tensor, \
                       (3) to DataContainer (stack=True)
    - padding_mask: (1) to tensor, (2) to DataContainer

    Args:
        ref_prefix (str): The prefix of key added to the second dict of input
            list. Defaults to 'ref'.
    """

    def __init__(self, ref_prefix='ref'):
        self.ref_prefix = ref_prefix

    def __call__(self, results):
        """Sequence Default formatting bundle call function.

        Args:
            results (list[dict]): List of two dicts.

        Returns:
            dict: The result dict contains the data that is formatted with
            default bundle. Each key in the second dict of the input list
            adds `self.ref_prefix` as prefix.
        """
        outs = []
        for _results in results:
            _results = self.default_format_bundle(_results)
            outs.append(_results)

        data = {}
        data.update(outs[0])
        for k, v in outs[1].items():
            data[f'{self.ref_prefix}_{k}'] = v

        print(f"{self.__class__.__name__} output: {data}")
        return data

    def default_format_bundle(self, results):
        """Transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
            default bundle.
        """
        if 'img' in results:
            img = results['img']

            if isinstance(img, torch.Tensor):
                # Already in tensor format from FormatTemporalInput
                pass
            elif len(img.shape) == 4:  # (T,H,W,C)
                img = np.ascontiguousarray(img.transpose(0, 3, 1, 2))  # (T,C,H,W)
            elif len(img.shape) == 3:  # (H,W,C)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))  # (C,H,W)
            
            # Convert to tensor if not already
            if not isinstance(img, torch.Tensor):
                img = to_tensor(img)
            
            results['img'] = DC(img, stack=True)
        if 'padding_mask' in results:
            results['padding_mask'] = DC(
                to_tensor(results['padding_mask'].copy()), stack=True)
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_instance_ids', 'gt_match_indices'
        ]:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        for key in ['img_metas', 'gt_masks']:
            if key in results:
                results[key] = DC(results[key], cpu_only=True)
        if 'gt_semantic_seg' in results:
            semantic_seg = results['gt_semantic_seg']
            if len(semantic_seg.shape) == 2:
                semantic_seg = semantic_seg[None, ...]
            else:
                semantic_seg = np.ascontiguousarray(
                    semantic_seg.transpose(3, 2, 0, 1))
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg']), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class VideoCollect(object):
    """Collect data from the loader relevant to the specific task.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str]): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('filename',
            'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
            'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg',
            'frame_id', 'is_video_data').
    """

    def __init__(self,
                 keys,
                 meta_keys=None,
                 default_meta_keys=('filename', 'ori_filename', 'ori_shape',
                                    'img_shape', 'pad_shape', 'scale_factor',
                                    'flip', 'flip_direction', 'img_norm_cfg',
                                    'frame_id', 'is_video_data')):
        self.keys = keys
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys, )
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

    def __call__(self, results):
        """Call function to collect keys in results.

        The keys in ``meta_keys`` and ``default_meta_keys`` will be converted
        to :obj:mmcv.DataContainer.

        Args:
            results (list[dict] | dict): List of dict or dict which contains
                the data to collect.

        Returns:
            list[dict] | dict: List of dict or dict that contains the
            following keys:

            - keys in ``self.keys``
            - ``img_metas``
        """
        results_is_dict = isinstance(results, dict)
        if results_is_dict:
            results = [results]
        outs = []
        for _results in results:
            _results = self._add_default_meta_keys(_results)
            _results = self._collect_meta_keys(_results)
            outs.append(_results)

        if results_is_dict:
            outs[0]['img_metas'] = DC(outs[0]['img_metas'], cpu_only=True)

        return outs[0] if results_is_dict else outs

    def _collect_meta_keys(self, results):
        """Collect `self.keys` and `self.meta_keys` from `results` (dict)."""
        data = {}
        # Handle both video sequence and single frame cases
        if 'images' in results:  # Video case
            # Collect data from first frame (for evaluation)
            frame_data = results['images'][0]
            for key in self.keys:
                if key in frame_data:
                    data[key] = frame_data[key]
        else:  # Single frame case
            for key in self.keys:
                if key in results:
                    data[key] = results[key]

        # Add metadata
        data['img_metas'] = {
            k: results.get(k, None) or results['images'][0].get(k, None)
            for k in self.meta_keys
        }
        '''
        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
            elif key in results['img_info']:
                img_meta[key] = results['img_info'][key]
        data['img_metas'] = img_meta
        for key in self.keys:
            data[key] = results[key]
        '''
        return data

    def _add_default_meta_keys(self, results):
        """Add default meta keys.

        We set default meta keys including `pad_shape`, `scale_factor` and
        `img_norm_cfg` to avoid the case where no `Resize`, `Normalize` and
        `Pad` are implemented during the whole pipeline.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            results (dict): Updated result dict contains the data to convert.
        """
        img = results['img']

        # Handle both tensor and numpy array inputs
        if isinstance(img, torch.Tensor):
            shape = img.shape[-2:]  # Get H,W from (T,C,H,W) or (C,H,W)
        else:
            shape = img.shape[:2] if len(img.shape) == 3 else img.shape[1:3]

        results.setdefault('pad_shape', img.shape)
        results.setdefault('scale_factor', 1.0)

        # Get channel count properly
        if isinstance(img, torch.Tensor):
            num_channels = img.shape[-3] if len(img.shape) == 4 else img.shape[0]
        else:
            num_channels = img.shape[-1] if len(img.shape) == 3 else 1
        
        results.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return results