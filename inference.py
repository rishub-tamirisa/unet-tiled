import torch
from os import walk
from unet1 import UNet
import torch.utils.data as data
import logging
from tqdm import tqdm, trange
import torch
import matplotlib.pyplot as plt
from utils.extra import IterDataset, tensor_to_tuple, Test_Train_Generator
from utils.ImageBuilder import ImageBuilder
import pickle

def save_true_validation_data(path: str, out_path=str, name=str):
    """
    Saves the true validation data to a pickle file
    """
    datapath = path + "/label/" + name + "_roisResized.pkl"
    obj = pickle.load(open(datapath, 'rb'))['resizedROIs']
    # print(obj.shape)
    plt.imshow(obj)
    plt.savefig(out_path + "/test/" + name + "_true.png")
    
    

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # assert False
    mypath = "/home/rshb/myspinner/kidney/data"
    filenames = next(walk(mypath), (None, None, []))[2] 

    split = 0.7
    # - - - - - - - HYPER PARAMETERS - - - - - - - -
    sparsity_threshold = 0.0
    tile_dim = 128

    # - - - - - - - INFERENCE LIST - - - - - - - -
    inference_list = ['NK2']

    datagen = Test_Train_Generator(dir_path=mypath, split=split, shuffle=True)

    dataset = IterDataset(generator=datagen.load_data_iter(
        inference=True, 
        tile_dim=tile_dim, 
        sparsity_threshold=sparsity_threshold,
        inference_list=inference_list))
    dataloader = data.DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = UNet(n_channels=1, n_classes=3)

    # load model from /home/rshb/myspinner/kidney/VisibleAligned/checkpoints
    epoch = 145
    net.load_state_dict(torch.load('/home/rshb/myspinner/kidney/VisibleAligned/checkpoints/unet_epoch'+str(epoch)+'.pth'))

    net.eval()
    net.to(device=device)

    image_builder = ImageBuilder(image_path='/home/rshb/myspinner/kidney/VisibleAligned/output/test', tile_dim=tile_dim)
    name = "Starting ... "
    with torch.no_grad():
        with tqdm(desc=f'Inference: ', unit=' tiles') as pbar1:
            for tile in dataloader:
                # print name only when name changes
                images = tile['image'].unsqueeze(1)
                true_masks = tile['mask']
                
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                masks_pred = net(images)
                pbar1.update(1)
                # store model output in image_builder
                output = masks_pred.detach().cpu()
                image_builder.add_to_storage(output, tensor_to_tuple(tile, 'index'), tile['name'][0], tensor_to_tuple(tile, 'image_shape'))
        
        for name in inference_list:
            
            # name = tile['name'][0]

            image_builder.build_and_save_image(name) 
            image_builder.reset_storage(name) 
            # clear storage as images are built TODO: make this happen as tiles are added, not when images are built 
            save_true_validation_data(path=mypath, out_path='/home/rshb/myspinner/kidney/VisibleAligned/output', name=name)

            # accuracy = pixelwise_accuracy(tile['mask'], image_builder._get_predicted_mask_from_output(output))

            print(name + " Saved ")


            

        
        





    