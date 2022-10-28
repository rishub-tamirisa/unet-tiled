import torch

from os import walk
import torch
import pickle
import numpy as np
import warnings


def _extract_tiles_generator(image, mask, tile_dim: int = 256, sparsity_threshold: float = 0.1, name: str = '', dense_multiplier: int = 1,training:bool=False,  validation: bool = False, inference:bool = False, shuffle: bool = False):
    '''
    Extracts tiles from the given image and mask.
    
    Args:
        image (np.ndarray): The image to extract the tiles from.
        mask (np.ndarray): The mask to extract the tiles from.
        tile_dim (int): The dimension of the tiles to be extracted from the images.
        sparsity_threshold (float): The threshold for the sparsity of the mask.
        name (str): The name of the image.
        dense_multiplier (int): The multiplier for the number of dense tiles to be extracted.
        training (bool): Whether to duplicate the tiles for the training data.
        validation (bool): Whether to duplicate the tiles for validation or not.
        inference (bool): Whether the mode is inference or not.
        shuffle (bool): Whether to shuffle the data or not.
    Yields:
        dict: A dictionary containing the image and mask tiles.
    '''
    assert not (validation and inference), 'validation and inference cannot be True at the same time'
    _i_idx = np.arange(0, image.shape[0] - tile_dim, tile_dim)
    _j_idx = np.arange(0, image.shape[1] - tile_dim, tile_dim)

    if shuffle:
        np.random.shuffle(_i_idx)
        np.random.shuffle(_j_idx)

    for i in _i_idx:
        for j in _j_idx:
            tile = image[i:i + tile_dim, j:j + tile_dim]
            mask_tile = mask[i:i + tile_dim, j:j + tile_dim]
            if (mask_tile.shape[0] ** 2) == 0:
                continue
            if inference: # inference mode first for faster inference (no need to calculate density)
                yield {'image': tile, 'mask': mask_tile, 'index': (i, j), 'name': name, 'image_shape': (image.shape[0], image.shape[1]), 'duplicate': False}
            
            density = np.count_nonzero(mask_tile) / (mask_tile.shape[0] ** 2)
            if density >= sparsity_threshold:
                if density > 0.3 and training:
                    for _ in range(dense_multiplier):
                        yield {'image': tile, 'mask': mask_tile, 'index': (i, j), 'name': name, 'image_shape': (image.shape[0], image.shape[1]), 'duplicate': True}
                if validation:
                    yield {'image': tile, 'mask': mask_tile, 'index': (i, j), 'name': name, 'image_shape': (image.shape[0], image.shape[1]), 'duplicate': False}

def _create_train_test_split(all_files:list=None, label_files:list=None, split:float=0.8, shuffle:bool=True):
    '''
    Parse files in all_files which have the format '(sample)(sample_id).pkl' into a dictionary with sample keys and sample_ids. sample and sample_id are all strings, but sample will contain no numbers and sample_id starts with a number. Label files have the format '(sample)(sample_id)_roisResized.pkl'
    
    ### Example **:
    
    AM8.pkl, AM9a.pkl, MC4.pkl, M10.pkl, and MC6.pkl should be parsed into the dictionary:
    
    {'MC' : ['4', '10', '6'], 'AM': ['8', '9a']}

    '''
    assert all_files is not None, 'all_files cannot be None'
    assert label_files is not None, 'label_files cannot be None'
    assert split > 0 and split < 1, 'split must be between 0 and 1'
    assert len(all_files) == len(label_files), 'all_files and label_files must have the same length'
    
    all_files_dict = {}
    for file in all_files:
        sample = ''.join([i for i in file if not i.isdigit()])[0:-4]
        sample_id = ''.join([i for i in file if i.isdigit()])
        if sample not in all_files_dict:
            all_files_dict[sample] = [sample_id]
        else:
            all_files_dict[sample].append(sample_id)
    
    if shuffle:
        for sample in all_files_dict:
            np.random.shuffle(all_files_dict[sample])

    train_files = []
    train_label_files = []
    
    for sample in all_files_dict:
        # assert len(all_files_dict[sample]) == len(label_files_dict[sample]), f'Number of files for sample {sample} does not match'
        num_files = len(all_files_dict[sample])
        num_train_files = int(num_files * split)
        train_files.extend([f'{sample}{id}.pkl' for id in all_files_dict[sample][:num_train_files]])
        train_label_files.extend([f'{sample}{id}_roisResized.pkl' for id in all_files_dict[sample][:num_train_files]])
    
    test_files = []
    test_label_files = []

    for sample in all_files_dict:
        # assert len(all_files_dict[sample]) == len(label_files_dict[sample]), f'Number of files for sample {sample} does not match'
        num_files = len(all_files_dict[sample])
        num_train_files = int(num_files * split)
        test_files.extend([f'{sample}{id}.pkl' for id in all_files_dict[sample][num_train_files:]])
        test_label_files.extend([f'{sample}{id}_roisResized.pkl' for id in all_files_dict[sample][num_train_files:]])

    return train_files, train_label_files, test_files, test_label_files

def _compute_data_from_path(dir_path:str=None, split:float=0.5, shuffle=True):
    '''
    Compute the data with a train/test split.

    ## Needs to be called
    Args:
        dir_path (str): The path to the directory containing the data.
        split (float): The percentage of the data to use for training.
        shuffle (bool): Whether or not to shuffle the data.
    Returns:
        train_data (list): A list of dictionaries containing the training data.
        test_data (list): A list of dictionaries containing the testing data.
    '''
    # issue warning if self.computed is more than 1
    TRAIN_PATH = dir_path + '/train/'
    LABEL_PATH = dir_path + '/label/'

    train_filenames = sorted(list(next(walk(TRAIN_PATH), (None, None, []))[2]))
    label_filenames = sorted(list(next(walk(LABEL_PATH), (None, None, []))[2]))

    train_files, train_label_files, test_files, test_label_files = _create_train_test_split(train_filenames, label_filenames, split, shuffle)

    train_files = [TRAIN_PATH + file for file in train_files]
    train_label_files = [LABEL_PATH + file for file in train_label_files]
    test_files = [TRAIN_PATH + file for file in test_files]
    test_label_files = [LABEL_PATH + file for file in test_label_files]

    return train_files, train_label_files, test_files, test_label_files

class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator

class Test_Train_Generator():

    def __init__(self, dir_path:str=None, split:float=0.5, shuffle=True):
        '''
        dir_path: path to directory containing all files
        split: float between 0 and 1, determines the split between training and testing data
        shuffle: bool, determines whether to shuffle the data
        '''
        self.path = dir_path
        self.split = split
        self.shuffle = shuffle

        self.train_files, self.train_label_files, self.test_files, self.test_label_files = _compute_data_from_path(self.path, self.split, self.shuffle)

    def length(self, mode:str='train'):
        '''
        Gets the length of the data in the given mode.

        Args:
            mode (str): The mode to return the length of. Must be 'train', 'test', or all.
        Returns:
            length (int): The length of the data.

        '''
        if mode == 'train':
            return len(self.train_files)
        elif mode == 'test':
            return len(self.test_files)
        elif mode == 'all':
            return len(self.train_files) + len(self.test_files)
        else:
            raise ValueError('mode must be either train, test, or all')

    def load_data_iter(self, dense_multiplier:int=1, training:bool=False, validation:bool=False, inference:bool=False, tile_dim: int = 256, sparsity_threshold: float = 0.1, inference_list: list = None):
        '''
        Loads the training data from the given directory path.
        
        Args:
            dir_path (str): The directory path to the training data.
            split (float): The percentage of the data to be used for training.
            dense_multiplier (int): The multiplier for the number of dense tiles to be extracted.
            training (bool): Whether to load the training data or the validation data.
            validation (bool): Whether to load the validation data or the training data.
            inference (bool): Whether the mode is inference or not.
            tile_dim (int): The dimension of the tiles to be extracted from the images.
            sparsity_threshold (float): The threshold for the sparsity of the mask.
            inference_list (list): The list of images to be used for inference.
        Returns:
            Generator: A generator that yields the training data.
        '''
        # assert only one of training, validation, inference is True
        assert int(training)+int(validation)+int(inference) == 1, 'only one of training, validation, inference can be True'
        if inference:
            assert inference_list is not None, 'inference_list cannot be None if inference is True'

        # get the data

        files = None
        label_files = None
        if training:
            files = self.train_files
            label_files = self.train_label_files
        elif validation:
            files = self.test_files
            label_files = self.test_label_files
        elif inference: # inference special behavior
            files = self.train_files + self.test_files
            label_files = self.train_label_files + self.test_label_files

        assert len(files) == len(label_files), 'Number of files does not match number of label files'
        _name_cutoff = self.path + '/train/' 
        for i, _ in enumerate(files):
            name = files[i][len(_name_cutoff):-4]
            if (inference and name not in inference_list) or (not inference and name in inference_list):
                continue
            
            train = torch.Tensor(pickle.load(open(files[i], 'rb'))['VisRegistered'])#)
            label = torch.Tensor(pickle.load(open(label_files[i], 'rb'))['resizedROIs'])

            for tile in _extract_tiles_generator(train, label, tile_dim, sparsity_threshold, name, dense_multiplier=dense_multiplier, training=training, validation=validation,inference = inference, shuffle=self.shuffle):
                yield tile

def pixelwise_accuracy(true_mask: torch.Tensor, predicted_mask: torch.Tensor):
    '''
    Calculates the pixelwise accuracy.
    
    Args:
        true_mask (torch.Tensor): The true mask.
        predicted_mask (torch.Tensor): The predicted mask.
    Returns:
        float: The pixelwise accuracy.
    '''
    return (true_mask == predicted_mask).sum().item() / (true_mask.shape[0] * true_mask.shape[1] * true_mask.shape[2])

def tensor_to_tuple(batch: torch.Tensor, key: str):
    '''
    Converts a tensor to a tuple.
    
    Args:
        tensor (torch.Tensor): The tensor to be converted.
    Returns:
        tuple: The converted tensor.
    '''
    return (int(batch[key][0][0]), int(batch[key][1][0]))

def normalize(tensor: torch.Tensor):
    '''
    Normalizes the given tensor.
    
    Args:
        tensor (torch.Tensor): The tensor to be normalized.
        batched (bool): Whether the tensor is batched or not.
    Returns:
        torch.Tensor: The normalized tensor.
    '''
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())
