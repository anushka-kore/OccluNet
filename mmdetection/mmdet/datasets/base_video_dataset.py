# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from collections import defaultdict
from typing import Any, List, Tuple

import mmengine.fileio as fileio
from mmengine.dataset import BaseDataset
from mmengine.logging import print_log

from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import DATASETS

from tqdm import tqdm


@DATASETS.register_module()
class BaseVideoDataset(BaseDataset):
    """Base video dataset for VID, MOT and VIS tasks."""

    METAINFO = {
        'classes' : ('Occlusion', )
    }

    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True
    
    def __init__(self, seq_len: int = None, *args, backend_args: dict = None, **kwargs):
        self.seq_len = seq_len  
        self.backend_args = backend_args
        super().__init__(*args, **kwargs)
       
    def get_frame_number(self, filename: str) -> int:
        """Extract frame number from filename."""
        base_name = osp.basename(filename)
        frame_num = base_name.split('_')[-1].split('.')[0]
        return int(frame_num)

    def get_sequence_name(self, filename: str) -> str:
        """Extract sequence name from filename."""
        return osp.basename(osp.dirname(filename))

    def load_data_list(self) -> Tuple[List[dict], List]:
        """Load annotations from an annotation file named as ``self.ann_file``."""
        with fileio.get_local_path(self.ann_file) as local_path:
            self.coco = COCO(local_path)
        
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)
        self.img_ids_with_ann = set()

        img_ids = self.coco.get_img_ids()
        total_ann_ids = []
        single_video_id = 100000
        data_list = []

        print(f"\n=== Loading Dataset ===")
        print(f"Total videos: {len(self.coco.dataset['videos'])}" if 'videos' in self.coco.dataset 
              else f"Total images: {len(self.coco.dataset['images'])}")
        print(f"Sequence length: {self.seq_len}")

        # First organize all frames by their sequence/video
        sequences = defaultdict(list)
        for img_id in img_ids:
            img_info = self.coco.load_imgs([img_id])[0]
            img_info['img_id'] = img_id
            sequence_name = self.get_sequence_name(img_info['file_name'])
            sequences[sequence_name].append(img_info)

        # Sort each sequence by frame number
        for seq_name in sequences:
            sequences[seq_name] = sorted(sequences[seq_name], 
                                       key=lambda x: self.get_frame_number(x['file_name']))

        # Now process each frame as a center frame and get its sequence
        for img_id in tqdm(img_ids, desc="Processing frames"):
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id
            
            # Get or assign video_id
            if 'video_id' not in raw_img_info:
                single_video_id += 1
                video_id = single_video_id
            else:
                video_id = raw_img_info['video_id']

            sequence_name = self.get_sequence_name(raw_img_info['file_name'])
            sequence_frames = sequences[sequence_name]
            current_frame_num = self.get_frame_number(raw_img_info['file_name'])
            
            # Find current frame index in sequence
            current_frame_idx = next(i for i, f in enumerate(sequence_frames) 
                                   if self.get_frame_number(f['file_name']) == current_frame_num)

            # Calculate start and end indices for the sequence
            half_window = (self.seq_len - 1) // 2
            start_idx = max(0, current_frame_idx - half_window)
            end_idx = min(len(sequence_frames), current_frame_idx + half_window + 1)
            
            # Adjust window if near sequence boundaries
            if end_idx - start_idx < self.seq_len:
                if start_idx == 0:
                    end_idx = min(len(sequence_frames), start_idx + self.seq_len)
                else:
                    start_idx = max(0, end_idx - self.seq_len)

            # Get the sequence of frames
            sequence_window = sequence_frames[start_idx:end_idx]
            
            # Build full paths for all frames in the sequence
            sequence_paths = []
            for frame in sequence_window:
                if self.data_prefix.get('img_path', None) is not None:
                    full_path = osp.join(self.data_prefix['img_path'], frame['file_name'])
                else:
                    full_path = frame['file_name']
                sequence_paths.append(full_path)

            # Load annotations for current frame only
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id], cat_ids=self.cat_ids)
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info(
                dict(raw_img_info=raw_img_info, raw_ann_info=raw_ann_info))

            if len(parsed_data_info['instances']) > 0:
                self.img_ids_with_ann.add(parsed_data_info['img_id'])

            # Create the data entry
            data_info = {
                'video_id': video_id,
                'images': sequence_paths,  # Full paths to all frames in the sequence
                'video_length': len(sequence_frames),
                'img_id': parsed_data_info['img_id'],
                'file_name': sequence_paths[len(sequence_paths)//2],  # Center frame path
                'width': parsed_data_info['width'],
                'height': parsed_data_info['height'],
                'frame_number': current_frame_num,
                'instances': parsed_data_info['instances'],
                'sequence_name': sequence_name
            }

            data_list.append(data_info)

        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(total_ann_ids), \
                   f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation to target format."""
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        data_info.update(img_info)
        
        if self.data_prefix.get('img_path', None) is not None:
            img_path = osp.join(self.data_prefix['img_path'], img_info['file_name'])
        else:
            img_path = img_info['file_name']
        data_info['img_path'] = img_path

        instances = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
                
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
                
            # Apply min_size filtering if specified in filter_cfg
            if hasattr(self, 'filter_cfg') and self.filter_cfg is not None:
                min_size = self.filter_cfg.get('min_size', 0)
                if w < min_size or h < min_size:
                    continue

            instance = {
                'ignore_flag': 1 if ann.get('iscrowd', False) else 0,
                'bbox': [x1, y1, x1 + w, y1 + h],
                'bbox_label': self.cat2label[ann['category_id']],
                'instance_id': ann.get('instance_id', i)
            }
            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']
            instances.append(instance)
            
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[int]:
        """Filter image annotations according to filter_cfg.

        Returns:
            list[int]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if not hasattr(self, 'filter_cfg') or self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', True)
        min_size = self.filter_cfg.get('min_size', 0)

        # Obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for class_id in self.cat_ids:
            ids_in_cat |= set(self.cat_img_map[class_id])
        
        valid_inds = []
        total_num = 0
        valid_num = 0
        
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            
            # Filter empty GT
            if filter_empty_gt and img_id not in self.img_ids_with_ann:
                continue
                
            # Filter small images
            if min(width, height) < min_size:
                continue
                
            valid_inds.append(i)
            valid_num += 1
            total_num += 1

        print_log(
            f'Filtered {total_num - valid_num} images, {valid_num} images remain.',
            logger='current')
            
        return [self.data_list[i] for i in valid_inds]
    
    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``."""
        data_info = self.get_data_info(idx)
        
        if self.test_mode:
            # In test mode, don't include instances
            data_info = copy.deepcopy(data_info)
            #data_info['instances'] = []
        
        data = self.pipeline(data_info)
        return data

    def get_cat_ids(self, index) -> List[int]:
        """Following image detection, we provide this interface function. Get
        category ids by video index and frame index.

        Args:
            index: The index of the dataset. It support two kinds of inputs:
                Tuple:
                    video_idx (int): Index of video.
                    frame_idx (int): Index of frame.
                Int: Index of video.

        Returns:
            List[int]: All categories in the image of specified video index
            and frame index.
        """
        if isinstance(index, tuple):
            assert len(
                index
            ) == 2, f'Expect the length of index is 2, but got {len(index)}'
            video_idx, frame_idx = index
            instances = self.get_data_info(
                video_idx)['images'][frame_idx]['instances']
            return [instance['bbox_label'] for instance in instances]
        else:
            cat_ids = []
            for img in self.get_data_info(index)['images']:
                for instance in img['instances']:
                    cat_ids.append(instance['bbox_label'])
            return cat_ids

    @property
    def num_all_imgs(self):
        """Get the number of all the images in this video dataset."""
        return sum(
            [len(self.get_data_info(i)['images']) for i in range(len(self))])

    def get_len_per_video(self, idx):
        """Get length of one video.

        Args:
            idx (int): Index of video.

        Returns:
            int (int): The length of the video.
        """
        return len(self.get_data_info(idx)['images'])
