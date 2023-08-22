import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from torchtext import data

from datasets.load_features import fill_missing_features, load_features_from_npy

#cfg is defined in load_features_from_npy in load_features.py
def caption_iterator(cfg, batch_size, phase): #this function is used to create a caption iterator for the train, val_1, val_2, and learned_props phases.
    print(f'Contructing caption_iterator for "{phase}" phase') # train, val_1, val_2, learned_props
    spacy_en = spacy.load('en') #load spacy model
    
    def tokenize_en(txt): #tokenize the captions divide words
        return [token.text for token in spacy_en.tokenizer(txt)] 
    
    CAPTION = data.ReversibleField( #ReversibleField is a subclass of Field that allows reversing a tokenized string.
        tokenize='spacy', init_token=cfg.start_token, eos_token=cfg.end_token, 
        pad_token=cfg.pad_token, lower=True, batch_first=True, is_target=True
    )
    INDEX = data.Field( #Field is a class that stores the vocabulary and numericalizes a given string.
        sequential=False, use_vocab=False, batch_first=True
    )
    
    # the order has to be the same as in the table
    fields = [
        ('video_id', None), 
        ('caption', CAPTION),
        ('start', None),
        ('end', None),
        ('duration', None),
        ('phase', None), # this is used to filter the dataset
        ('idx', INDEX), # this is used to get the features from the dataset
    ]

    dataset = data.TabularDataset(
        path=cfg.train_meta_path, format='tsv', skip_header=True, fields=fields,
    ) #TabularDataset is a class that is used to load data from a csv file.
    CAPTION.build_vocab(dataset.caption, min_freq=cfg.min_freq_caps, vectors=cfg.word_emb_caps) #build_vocab is used to build a vocabulary from the captions in the dataset.
    train_vocab = CAPTION.vocab
    
    if phase == 'val_1': #if phase is val_1, the dataset is loaded from the val_1_meta_path.
        dataset = data.TabularDataset(path=cfg.val_1_meta_path, format='tsv', skip_header=True, fields=fields) 
    elif phase == 'val_2': #if phase is val_2, the dataset is loaded from the val_2_meta_path.
        dataset = data.TabularDataset(path=cfg.val_2_meta_path, format='tsv', skip_header=True, fields=fields)
    elif phase == 'learned_props': #if phase is learned_props, the dataset is loaded from the val_prop_meta_path.
        dataset = data.TabularDataset(path=cfg.val_prop_meta_path, format='tsv', skip_header=True, fields=fields)

    # sort_key = lambda x: data.interleave_keys(len(x.caption), len(y.caption))
    datasetloader = data.BucketIterator(dataset, batch_size, sort_key=lambda x: 0, 
                                        device=torch.device(cfg.device), repeat=False, shuffle=True)
    return train_vocab, datasetloader


class I3DFeaturesDataset(Dataset): #this class is used to create a dataset for the video features.
    
    def __init__(self, features_path, feature_name, meta_path, device, pad_idx, get_full_feat, cfg):    #features_path is the path to the video features, feature_name is the name of the video feature, meta_path is the path to the meta file, device is the device to be used, pad_idx is the index of the padding token, get_full_feat is a boolean value that indicates whether to get the full feature or not, and cfg is the configuration file.
        self.cfg = cfg 
        self.features_path = features_path
        self.feature_name = f'{feature_name}_features' 
        self.feature_names_list = [self.feature_name]
        self.device = device #cpu/gpu
        self.dataset = pd.read_csv(meta_path, sep='\t') #read the meta file which we just created in load features.
        self.pad_idx = pad_idx #padding token index     
        self.get_full_feat = get_full_feat #boolean value to get full feature or not
        
        if self.feature_name == 'i3d_features': 
            self.feature_size = 1024
        else:
            raise Exception(f'Inspect: "{self.feature_name}"')
    
    
        #BELOW GETiTEM FUNC
        #It prepares empty lists to hold video IDs, captions, start times, end times, and video feature stacks.
        #For each index in the given indices:
        #Extract the video's information (like ID, caption, start, end, duration) from the metadata.
        #Load video feature stacks using the load_features_from_npy function with provided parameters.
        #Separate the RGB and flow feature stacks.
        #Check if both RGB and flow stacks are either both None or both not None.
        #If both stacks are None, replace them with zeros.
        #Append all information for this index to the respective lists.
        #Use pad_sequence to pad the RGB and flow feature stacks in a batch-friendly manner.
        #Convert start and end times into tensors and shape them correctly.
        #Create a dictionary containing all the processed data, including video IDs, captions, start times, end times, and feature stacks.
    def __getitem__(self, indices): #retrieve items from the dataset when an index (or a batch of indices) is provided. It processes the data and returns it in a dictionary.
        video_ids, captions, starts, ends, vid_stacks_rgb, vid_stacks_flow = [], [], [], [], [], []

        for idx in indices:
            idx = idx.item()
            video_id, caption, start, end, duration, _, _ = self.dataset.iloc[idx]
            
            stack = load_features_from_npy(
                self.cfg, self.feature_names_list, video_id, start, end, duration, 
                self.pad_idx, self.get_full_feat
            )

            vid_stack_rgb, vid_stack_flow = stack['rgb'], stack['flow']
            
            # either both None or both are not None (Boolean Equivalence)
            both_are_None = vid_stack_rgb is None and vid_stack_flow is None
            none_is_None = vid_stack_rgb is not None and vid_stack_flow is not None
            assert both_are_None or none_is_None
            
            # # sometimes stack is empty after the filtering. we replace it with noise
            if both_are_None:
                # print(f'RGB and FLOW are None. Zero (1, D) @: {video_id}')
                vid_stack_rgb = fill_missing_features('zero', self.feature_size)
                vid_stack_flow = fill_missing_features('zero', self.feature_size)
    
            # append info for this index to the lists
            video_ids.append(video_id)
            captions.append(caption)
            starts.append(start)
            ends.append(end)
            vid_stacks_rgb.append(vid_stack_rgb)
            vid_stacks_flow.append(vid_stack_flow)
            
        vid_stacks_rgb = pad_sequence(vid_stacks_rgb, batch_first=True, padding_value=self.pad_idx)
        vid_stacks_flow = pad_sequence(vid_stacks_flow, batch_first=True, padding_value=0)
                
        starts = torch.tensor(starts).unsqueeze(1)
        ends = torch.tensor(ends).unsqueeze(1)

        batch_dict = {
            'video_ids': video_ids,
            'captions': captions,
            'starts': starts.to(self.device),
            'ends': ends.to(self.device),
            'feature_stacks': {
                'rgb': vid_stacks_rgb.to(self.device),
                'flow': vid_stacks_flow.to(self.device),
            }
        }
        
        return batch_dict

    def __len__(self):
        return len(self.dataset)
    
class VGGishFeaturesDataset(Dataset):
    
    def __init__(self, features_path, feature_name, meta_path, device, pad_idx, get_full_feat, cfg):
        self.cfg = cfg
        self.features_path = features_path
        self.feature_name = 'vggish_features'
        self.feature_names_list = [self.feature_name]
        self.device = device
        self.dataset = pd.read_csv(meta_path, sep='\t')
        self.pad_idx = pad_idx
        self.get_full_feat = get_full_feat
        self.feature_size = 128
            #BELOW GETiTEM FUNC
                #This method is used to retrieve items from the dataset when an index (or a batch of indices) is provided. It processes the audio features and returns them in a dictionary, similar to the previous class.
                #It prepares empty lists to hold video IDs, captions, start times, end times, and audio feature stacks.
                #For each index in the given indices:
                #Extract the video's information (like ID, caption, start, end, duration) from the metadata.
                #Load audio feature stacks using the load_features_from_npy function with provided parameters.
                #Separate the audio feature stack.
                #Check if the audio stack is None and replace it with zeros if needed.
                #Append all information for this index to the respective lists.
                #Use pad_sequence to pad the audio feature stacks in a batch-friendly manner.
                #Convert start and end times into tensors and shape them correctly.
                #Create a dictionary containing all the processed data, including video IDs, captions, start times, end times, and feature stacks.        
    def __getitem__(self, indices):
        video_ids, captions, starts, ends, aud_stacks = [], [], [], [], []

        # [3]
        for idx in indices:
            idx = idx.item()
            video_id, caption, start, end, duration, _, _ = self.dataset.iloc[idx]
            
            stack = load_features_from_npy(
                self.cfg, self.feature_names_list, video_id, start, end, duration,
                self.pad_idx, self.get_full_feat
            )
            aud_stack = stack['audio']
            
            # sometimes stack is empty after the filtering. we replace it with noise
            if aud_stack is None:
                # print(f'VGGish is None. Zero (1, D) @: {video_id}')
                aud_stack = fill_missing_features('zero', self.feature_size)
    
            # append info for this index to the lists
            video_ids.append(video_id)
            captions.append(caption)
            starts.append(start)
            ends.append(end)
            aud_stacks.append(aud_stack)
            
        # [4] see ActivityNetCaptionsDataset.__getitem__ documentation
        aud_stacks = pad_sequence(aud_stacks, batch_first=True, padding_value=self.pad_idx)
                
        starts = torch.tensor(starts).unsqueeze(1)
        ends = torch.tensor(ends).unsqueeze(1)

        batch_dict = {
            'video_ids': video_ids,
            'captions': captions,
            'starts': starts.to(self.device),
            'ends': ends.to(self.device),
            'feature_stacks': {
                'audio': aud_stacks.to(self.device),
            }
        }

        return batch_dict

    def __len__(self):
        return len(self.dataset)
    
    
    
     # I think you get the trend numaan if you still need to write it down you are down bad
class AudioVideoFeaturesDataset(Dataset):
    
    def __init__(self, video_features_path, video_feature_name, audio_features_path, 
                 audio_feature_name, meta_path, device, pad_idx, get_full_feat, cfg):
        self.cfg = cfg
        self.video_features_path = video_features_path
        self.video_feature_name = f'{video_feature_name}_features'
        self.audio_features_path = audio_features_path
        self.audio_feature_name = f'{audio_feature_name}_features'
        self.feature_names_list = [self.video_feature_name, self.audio_feature_name]
        self.device = device
        self.dataset = pd.read_csv(meta_path, sep='\t')
        self.pad_idx = pad_idx
        self.get_full_feat = get_full_feat
        
        if self.video_feature_name == 'i3d_features':
            self.video_feature_size = 1024
        else:
            raise Exception(f'Inspect: "{self.video_feature_name}"')
            
        if self.audio_feature_name == 'vggish_features':
            self.audio_feature_size = 128
        else:
            raise Exception(f'Inspect: "{self.audio_feature_name}"')
            
    
    def __getitem__(self, indices):
        video_ids, captions, starts, ends = [], [], [], []
        vid_stacks_rgb, vid_stacks_flow, aud_stacks = [], [], []
        
        # [3]
        for idx in indices:
            idx = idx.item()
            video_id, caption, start, end, duration, _, _ = self.dataset.iloc[idx]
            
            stack = load_features_from_npy(
                self.cfg, self.feature_names_list,
                video_id, start, end, duration, self.pad_idx, self.get_full_feat
            )
            vid_stack_rgb, vid_stack_flow, aud_stack = stack['rgb'], stack['flow'], stack['audio']

            # either both None or both are not None (Boolean Equivalence)
            both_are_None = vid_stack_rgb is None and vid_stack_flow is None
            none_is_None = vid_stack_rgb is not None and vid_stack_flow is not None
            assert both_are_None or none_is_None

            # sometimes vid_stack and aud_stack are empty after the filtering. 
            # we replace it with noise.
            # tied with assertion above
            if (vid_stack_rgb is None) and (vid_stack_flow is None):
                # print(f'RGB and FLOW are None. Zero (1, D) @: {video_id}')
                vid_stack_rgb = fill_missing_features('zero', self.video_feature_size)
                vid_stack_flow = fill_missing_features('zero', self.video_feature_size)
            if aud_stack is None:
                # print(f'Audio is None. Zero (1, D) @: {video_id}')
                aud_stack = fill_missing_features('zero', self.audio_feature_size)

            # append info for this index to the lists
            video_ids.append(video_id)
            captions.append(caption)
            starts.append(start)
            ends.append(end)
            vid_stacks_rgb.append(vid_stack_rgb)
            vid_stacks_flow.append(vid_stack_flow)
            aud_stacks.append(aud_stack)
            
        # [4] see ActivityNetCaptionsDataset.__getitem__ documentation
        # rgb is padded with pad_idx; flow is padded with 0s: expected to be summed later
        vid_stacks_rgb = pad_sequence(vid_stacks_rgb, batch_first=True, padding_value=self.pad_idx)
        vid_stacks_flow = pad_sequence(vid_stacks_flow, batch_first=True, padding_value=0)
        aud_stacks = pad_sequence(aud_stacks, batch_first=True, padding_value=self.pad_idx)

        starts = torch.tensor(starts).unsqueeze(1)
        ends = torch.tensor(ends).unsqueeze(1)
                
        batch_dict = {
            'video_ids': video_ids,
            'captions': captions,
            'starts': starts.to(self.device),
            'ends': ends.to(self.device),
            'feature_stacks': {
                'rgb': vid_stacks_rgb.to(self.device),
                'flow': vid_stacks_flow.to(self.device),
                'audio': aud_stacks.to(self.device),
            }
        }

        return batch_dict
        
    def __len__(self):
        return len(self.dataset)


class ActivityNetCaptionsDataset(Dataset):
    
    def __init__(self, cfg, phase, get_full_feat):

        '''
            For the doc see the __getitem__.
        '''
        self.cfg = cfg
        self.phase = phase
        self.get_full_feat = get_full_feat

        self.feature_names = f'{cfg.video_feature_name}_{cfg.audio_feature_name}'
        
        if phase == 'train':
            self.meta_path = cfg.train_meta_path
            self.batch_size = cfg.train_batch_size
        elif phase == 'val_1':
            self.meta_path = cfg.val_1_meta_path
            self.batch_size = cfg.inference_batch_size
        elif phase == 'val_2':
            self.meta_path = cfg.val_2_meta_path
            self.batch_size = cfg.inference_batch_size
        elif phase == 'learned_props':
            self.meta_path = cfg.val_prop_meta_path
            self.batch_size = cfg.inference_batch_size
        else:
            raise NotImplementedError

        # caption dataset *iterator*
        self.train_vocab, self.caption_loader = caption_iterator(cfg, self.batch_size, self.phase)
        
        self.trg_voc_size = len(self.train_vocab)
        self.pad_idx = self.train_vocab.stoi[cfg.pad_token]
        self.start_idx = self.train_vocab.stoi[cfg.start_token]
        self.end_idx = self.train_vocab.stoi[cfg.end_token]
            
        if cfg.modality == 'video':
            self.features_dataset = I3DFeaturesDataset(
                cfg.video_features_path, cfg.video_feature_name, self.meta_path, 
                torch.device(cfg.device), self.pad_idx, self.get_full_feat, cfg
            )
        elif cfg.modality == 'audio':
            self.features_dataset = VGGishFeaturesDataset(
                cfg.audio_features_path, cfg.audio_feature_name, self.meta_path, 
                torch.device(cfg.device), self.pad_idx, self.get_full_feat, cfg
            )
        elif cfg.modality == 'audio_video':
            self.features_dataset = AudioVideoFeaturesDataset(
                cfg.video_features_path, cfg.video_feature_name, cfg.audio_features_path, 
                cfg.audio_feature_name, self.meta_path, torch.device(cfg.device), self.pad_idx, 
                self.get_full_feat, cfg
            )
        else:
            raise Exception(f'it is not implemented for modality: {cfg.modality}')
            
        # initialize the caption loader iterator
        self.caption_loader_iter = iter(self.caption_loader)
        
    def __getitem__(self, dataset_index):
        caption_data = next(self.caption_loader_iter)
        to_return = self.features_dataset[caption_data.idx]
        to_return['caption_data'] = caption_data

        return to_return

    def __len__(self):
        return len(self.caption_loader)
    
    def update_iterator(self):
        '''This should be called after every epoch'''
        self.caption_loader_iter = iter(self.caption_loader)
        
    def dont_collate(self, batch):
        return batch[0]
