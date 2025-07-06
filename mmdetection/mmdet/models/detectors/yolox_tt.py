import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmdet.registry import MODELS
from .yolox import YOLOX
from .timesformer import TimeSformerForFeatures
from .single_stage import SingleStageDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from typing import Optional, List, Tuple, Union
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from .temporal_transformer import TemporalTransformer



@MODELS.register_module()
class YOLOXTemporalTransformer(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 temporal_cfg: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        self.temporal_cfg = TemporalTransformer(input_dim=neck.out_channels, model_dim=neck.out_channels, num_heads=1, num_layers=4)

    def extract_feat(self, batch_inputs: torch.Tensor):
        """Extract features with temporal processing.
        
        Args:
            batch_inputs (Tensor): Shape (B, T, C, H, W) 
        """
                
        #print(f"batch_inputs: {batch_inputs.shape}")
        squeezed_input = batch_inputs.view(-1, *batch_inputs.shape[2:])  # (B*T, C, H, W)

        #print(f"[squeezed] batch_inputs shape: {squeezed_input.shape}")
        
        # Backbone + Neck processing
        backbone_feats = self.backbone(squeezed_input)
        neck_feats = self.neck(backbone_feats)  # (p3, p4, p5)
        
        #print(f"neck feats: {neck_feats}")

        # Process each FPN level
        in_features = list(neck_feats)
        T = batch_inputs.shape[1]
        #print(f"in_features: {in_features}")
        out_features = []
        for feature_idx, feature in enumerate(in_features):
            # unsqueeze feature
            BT, C, H, W = feature.shape
            feature = feature.view(BT//T, T, C, H, W)
            # TODO: padding
            feature = feature.permute(0, 3, 4, 1, 2)
            feature = feature.reshape(-1, T, C)  # BHWxTxC
            
            '''apply temporal to each layer feature'''
            #print(f"feature shape before temporal: {feature.shape}")
            out_feature = self.temporal_cfg(feature)
            #print(f"feature shape after temporal: {out_feature.shape}")
            out_feature = out_feature.view(BT//T, H, W, C).permute(0, 3, 1, 2)
            out_features.append(out_feature)

        #print(f"out_features: {out_features}")
        #print(f"out_features shape: {out_features.shape}")
        return out_features


    def loss(self, batch_inputs: torch.Tensor,
             batch_data_samples: Union[list, dict],
             **kwargs) -> Union[dict, tuple]:
        x = self.extract_feat(batch_inputs)
        return self.bbox_head.loss(x, batch_data_samples, **kwargs)

    def predict(self, batch_inputs, batch_data_samples, **kwargs):
        x = self.extract_feat(batch_inputs)

        # Get predictions
        raw_preds = self.bbox_head.predict(x, batch_data_samples, **kwargs)

        # Convert to standard format
        results = []
        for pred in raw_preds:
            result = DetDataSample()
            result.pred_instances = pred
            if batch_data_samples and len(batch_data_samples) > 0:
                ref_sample = batch_data_samples[0][0] if isinstance(batch_data_samples[0], (list, tuple)) else batch_data_samples[0]
                result.img_id = ref_sample.get('img_id', 0)
                result.file_name = ref_sample.get('file_name')
                result.ori_shape = ref_sample.get('ori_shape', (640, 640))
                result.instances = ref_sample.get('gt_instances')
            results.append(result)
        
        return results
       

@MODELS.register_module()
class YOLOXTSF(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 temporal_cfg: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        self.temporal_cfg = TemporalTransformer(input_dim=neck.out_channels, model_dim=neck.out_channels, num_heads=1, num_layers=4)

    def extract_feat(self, batch_inputs: torch.Tensor):
        """Extract features with temporal processing.
        
        Args:
            batch_inputs (Tensor): Shape (B, T, C, H, W) 
        """
                
        #print(f"batch_inputs: {batch_inputs.shape}")
        squeezed_input = batch_inputs.view(-1, *batch_inputs.shape[2:])  # (B*T, C, H, W)

        #print(f"[squeezed] batch_inputs shape: {squeezed_input.shape}")
        
        # Backbone + Neck processing
        backbone_feats = self.backbone(squeezed_input)
        neck_feats = self.neck(backbone_feats)  # (p3, p4, p5)
        
        #print(f"neck feats: {neck_feats}")

        # Process each FPN level
        in_features = list(neck_feats)
        T = batch_inputs.shape[1]
        #print(f"in_features: {in_features}")
        out_features = []
        for feature_idx, feature in enumerate(in_features):
            # unsqueeze feature
            BT, C, H, W = feature.shape
            feature = feature.view(BT//T, T, C, H, W)
            # TODO: padding
            feature = feature.permute(0, 3, 4, 1, 2)
                        
            '''apply temporal to each layer feature'''
            #print(f"feature shape before temporal: {feature.shape}")
            out_feature = self.temporal_cfg(feature.reshape(-1, T, C), spatial_shape=(H,W))
            #print(f"feature shape after temporal: {out_feature.shape}")
            out_feature = out_feature.view(BT//T, H, W, C).permute(0, 3, 1, 2)
            out_features.append(out_feature)

        #print(f"out_features: {out_features}")
        #print(f"out_features shape: {out_features.shape}")
        return out_features


    def loss(self, batch_inputs: torch.Tensor,
             batch_data_samples: Union[list, dict],
             **kwargs) -> Union[dict, tuple]:
        x = self.extract_feat(batch_inputs)
        return self.bbox_head.loss(x, batch_data_samples, **kwargs)

    def predict(self, batch_inputs, batch_data_samples, **kwargs):
        x = self.extract_feat(batch_inputs)

        # Get predictions
        raw_preds = self.bbox_head.predict(x, batch_data_samples, **kwargs)

        # Convert to standard format
        results = []
        for pred in raw_preds:
            result = DetDataSample()
            result.pred_instances = pred
            if batch_data_samples and len(batch_data_samples) > 0:
                ref_sample = batch_data_samples[0][0] if isinstance(batch_data_samples[0], (list, tuple)) else batch_data_samples[0]
                result.img_id = ref_sample.get('img_id', 0)
                result.file_name = ref_sample.get('file_name')
                result.ori_shape = ref_sample.get('ori_shape', (640, 640))
                result.instances = ref_sample.get('gt_instances')
            results.append(result)
        
        return results