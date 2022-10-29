import torch
import matplotlib.pyplot as plt
import numpy as np

class ImageBuilder:
    def __init__(self, image_path: str='myspinner/kidney/VisibleAligned/output/train', tile_dim: int=256):
        '''
        Initializes the ImageBuilder class. 
        
        Args:
            image_path (str): The path to the image.
        
        '''
        self.image_path = image_path
        self.tile_dim = tile_dim
        # define storage dictionary that stores keys as image names and values as tiles
        self.storage = {}
    
    def set_tile_dim(self, tile_dim: int):
        '''
        Sets the tile dimension.
        
        Args:
            tile_dim (int): The tile dimension.
        '''
        self.tile_dim = tile_dim

    def add_to_storage(self, output: torch.Tensor, index: tuple[int], name: str='', image_shape=None):
        '''
        Adds the given tile to the storage dictionary.
        
        Args:
            output (torch.Tensor): The tile to be added.
            index (tuple(int)): The index of the tile.
            name (str): The name of the image.
            image_shape (tuple(int)): The shape of the image.
        '''
        if name not in self.storage.keys():
            self.storage[name] = {}
        
        self.storage[name][index] = self._get_predicted_mask_from_output(output)
        if self.storage[name].get('image_shape') is None:
            self.storage[name]['image_shape'] = image_shape

    def reset_storage(self, name:str):
        '''
        Resets the storage dictionary for the given `name` key.
        
        Args:
            name (str): The name of the image.
        '''
        self.storage[name] = {}

    def build_and_save_image(self, name: str):
        '''
        Builds the image from the tiles in the storage dictionary.
        
        Args:
            name (str): The name of the image.
        Returns:
            np.ndarray: The image.
        '''
        assert name in self.storage, 'Image {name} not found in storage.'
        image = self.storage[name]
        # get the max index for each dimension
        max_i = max([i[0] for i in image.keys() if i != 'image_shape'])
        max_j = max([i[1] for i in image.keys() if i != 'image_shape'])
        # create an empty array of the correct size
        image_array = np.zeros(self.storage[name]['image_shape'])
        # fill the array with the tiles
        for index, _tile in image.items():
            if index != 'image_shape' and index != 'duplicate':
                tile = torch.squeeze(_tile)
                assert tile.shape == (self.tile_dim, self.tile_dim), f'Expected tile shape ({self.tile_dim}, {self.tile_dim}), got {tile.shape}.'
                image_array[index[0]:index[0]+self.tile_dim, index[1]:index[1]+self.tile_dim] = tile
        
        plt.imshow(image_array)
        plt.savefig(self.image_path + '/' + name + '.png')
        # return image_array

    def _get_predicted_mask_from_output(self, output: torch.Tensor):
        '''
        Gets the predicted mask from the output.
        
        Args:
            output (torch.Tensor): The output of the model.
            softmax (bool): Whether to apply softmax to the output.
        Returns:
            torch.Tensor: The predicted mask.
        '''
        output = torch.nn.functional.softmax(output, dim=1)
        return output.argmax(dim=1)