import os

import numpy as np
import torch
import torch.nn.functional as F

# I think you get the trend numaan if you still need to READ THESE COMMENTS AFTER TRAINING 16 times then you are a lost cause

#hope these comments make sense to you 
def fill_missing_features(method, feature_size): #this is base function ,benchmarked in both datasets.py
    if method == 'random': #If method is 'random', the function returns a tensor of shape (1, feature_size) filled with random values between 0 and 1 using torch.rand.
        #numaan a tensor is a multi-dimensional matrix containing elements of a single data type.
        return torch.rand(1, feature_size)
    elif method == 'zero': #If method is 'zero', the function returns a tensor of shape (1, feature_size) filled with zeros using torch.zeros.
        return torch.zeros(1, feature_size).float()

#feature indexes 
def crop_a_segment(feature, start, end, duration): #that is used to crop a segment from a given feature matrix.
    S, D = feature.shape #rows and columns
    start_quantile = start / duration # These are the start and end times (in terms of the segment index) of the segment you want to crop. These values indicate the range of segments you want to extract from the feature matrix.
    end_quantile = end / duration #decide if to crop?
    start_idx = int(S * start_quantile) #indexes
    end_idx = int(S * end_quantile)
    # case for if too small indexes
    if start_idx == end_idx:
        # if the small segment occurs in the end of a video
        # [S:S] -> [S-1:S]
        if start_idx == S:
            start_idx -= 1
        # [S:S] -> [S:S+1]
        else:
            end_idx += 1
    feature = feature[start_idx:end_idx, :]

#worst case    
    if len(feature) == 0:
        return None
    else:
        return feature


def pad_segment(feature, max_feature_len, pad_idx): # pads a given feature matrix to a specified length.
    S, D = feature.shape
    assert S <= max_feature_len #assert is used to check if a given condition is true or not. If it is true, nothing happens. But if it's false, an AssertionError is raised.
    # pad
    l, r, t, b = 0, 0, 0, max_feature_len - S   #left,right,top,bottom
    feature = F.pad(feature, [l, r, t, b], value=pad_idx) #F.pad is used to pad a tensor. The first argument is the tensor to be padded, the second argument is a list of four values that indicate the number of elements to be padded to the left, right, top, and bottom of the tensor, and the third argument is the value to be padded.
    return feature


def load_features_from_npy(cfg, feature_names_list, video_id, start, end, duration,
                           pad_idx, get_full_feat=False): #loads the features from the .npy files and returns a dictionary containing the features.
    supported_feature_names = {'i3d_features', 'vggish_features'}   #set of features
    assert isinstance(feature_names_list, list) 
    assert len(feature_names_list) > 0 
    assert set(feature_names_list).issubset(supported_feature_names)

    stacks = {} #load features
    if get_full_feat:
        stacks['orig_feat_length'] = {} #original feature length

    if 'vggish_features' in feature_names_list: 
        try:
            stack_vggish = np.load(os.path.join(cfg.audio_features_path, f'{video_id}.npy')) #load audio features
            stack_vggish = torch.from_numpy(stack_vggish).float() #convert to tensor

            if get_full_feat:
                stacks['orig_feat_length']['audio'] = stack_vggish.shape[0] 
                stack_vggish = pad_segment(stack_vggish, cfg.pad_feats_up_to['audio'], pad_idx)
            else:
                stack_vggish = crop_a_segment(stack_vggish, start, end, duration)
        except FileNotFoundError:
            stack_vggish = None
        stacks['audio'] = stack_vggish
    # not elif
    if 'i3d_features' in feature_names_list:
        try:
            stack_rgb = np.load(os.path.join(cfg.video_features_path, f'{video_id}_rgb.npy'))
            stack_flow = np.load(os.path.join(cfg.video_features_path, f'{video_id}_flow.npy'))
            stack_rgb = torch.from_numpy(stack_rgb).float()
            stack_flow = torch.from_numpy(stack_flow).float()

            assert stack_rgb.shape == stack_flow.shape
            if get_full_feat:
                stacks['orig_feat_length']['rgb'] = stack_rgb.shape[0]
                stacks['orig_feat_length']['flow'] = stack_flow.shape[0]
                stack_rgb = pad_segment(stack_rgb, cfg.pad_feats_up_to['video'], pad_idx)
                stack_flow = pad_segment(stack_flow, cfg.pad_feats_up_to['video'], pad_idx=0)
            else:
                stack_rgb = crop_a_segment(stack_rgb, start, end, duration)
                stack_flow = crop_a_segment(stack_flow, start, end, duration)
        except FileNotFoundError:
            stack_rgb = None
            stack_flow = None
        stacks['rgb'] = stack_rgb
        stacks['flow'] = stack_flow
    if 'i3d_features' not in feature_names_list and 'vggish_features' not in feature_names_list:
        raise Exception(f'This methods is not implemented for {feature_names_list}')

    return stacks

