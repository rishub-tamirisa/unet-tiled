import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

import torch
import logging
from os import walk
from tqdm import tqdm
import pickle
from unet1 import UNet, dice_loss
from utils.extra import IterDataset, pixelwise_accuracy, Test_Train_Generator, normalize
from utils.ImageBuilder import ImageBuilder
import torch.distributed as dist

def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://"  # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()

def train_net(net,
              device,
              train_dataloader,
              test_dataloader,
              n_channels,
              n_classes,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 2e-4,
              save_checkpoint: bool = True,
              tile_dim: int = 256,
              sparsity_threshold: float = 0.1,
              do_validation: bool = False,
              ):
    logging.info(f'Network:\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n')
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Tile dimension:  {tile_dim}
        Tile Threshold:   {sparsity_threshold}
        Do validation:   {do_validation}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, verbose=True, factor=0.1) 
    loss_function = torch.nn.CrossEntropyLoss()
    global_step = 0
    image_builder = ImageBuilder(image_path='/home/rshb/myspinner/kidney/VisibleAligned/output/train', tile_dim=tile_dim)
                        # randomly apply the color jitter to the tile
    
    # 5. Begin training
    loss_metrics = []
    for epoch in range(1, epochs+1):
        avg_acc = 0
        local_step = 0
        net.train()
        epoch_loss = 0
        avg_dice = 0
        with tqdm(desc=f'Epoch {epoch}/{epochs}', unit=' tiles') as pbar:
            # train_dataloader.sampler.set_epoch(epoch)
            for batch in train_dataloader:

                images = normalize(batch['image'].unsqueeze(1))#.permute(0,1))
                true_masks = batch['mask']
                assert images.shape[1] == n_channels, \
                    f'Network has been defined with {n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                optimizer.zero_grad()
                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                masks_pred = net(images)

                d_loss = dice_loss(torch.nn.functional.softmax(masks_pred, dim=1), torch.nn.functional.one_hot(
                    true_masks, n_classes).permute(0, 3, 1, 2).float(), multiclass=True)

                loss =  loss_function(masks_pred, true_masks) #+ d_loss
                avg_dice += ( 1- d_loss.item())
                output = masks_pred.detach().cpu()

                accuracy = pixelwise_accuracy(batch['mask'], image_builder._get_predicted_mask_from_output(output))

                loss.backward()
                optimizer.step()
                avg_acc += accuracy

                pbar.update(images.shape[0])
                global_step += 1
                local_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(
                    **{' loss (batch)': loss.item(), 'accuracy': (avg_acc) / (local_step), 'dice coeff': (avg_dice) / (local_step)})
                del images, true_masks, masks_pred

        if do_validation:
            net.eval()
            val_loss = 0
            with tqdm(desc=f'(VAL) Epoch {epoch}/{epochs}', unit=' tiles') as pbar1:
                for batch in test_dataloader:
                    with torch.no_grad():
                        images = batch['image'].unsqueeze(1)
                        true_masks = batch['mask']
                        images = images.to(device=device, dtype=torch.float32)
                        true_masks = true_masks.to(device=device, dtype=torch.long)
                        masks_pred = net(images)
                        d_loss = dice_loss(torch.nn.functional.softmax(masks_pred, dim=1), torch.nn.functional.one_hot(
                            true_masks, net.n_classes).permute(0, 3, 1, 2).float(), multiclass=True)
                        loss = loss_function(masks_pred, true_masks) + d_loss
                        val_loss += loss.item()
                        pbar1.update(images.shape[0])
                        pbar1.set_postfix(**{batch['name'][0] + ' loss (batch)': loss.item()})
                        del images, true_masks, masks_pred

            loss_metrics.append((epoch_loss, val_loss))

            net.train()
            scheduler.step(val_loss)

            logging.info(f'(VAL) Epoch {epoch}/{epochs} batch {local_step} - loss: {epoch_loss} - val_loss: {val_loss}\n')
        if save_checkpoint and local_step % 20 == 0:
            # save loss metrics to pickle file
            if not do_validation and local_step != 0:
                loss_metrics.append((avg_dice / local_step, epoch_loss))
            with open('train_metrics.pkl', 'wb') as f:
                pickle.dump(loss_metrics, f)
            torch.save(net.state_dict(), f'checkpoints/unet_epoch{epoch}.pth')
            # logging.info(f'Checkpoint {epoch} saved !\n')
    
    
if __name__ == "__main__":
    '''
    torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py
    '''
    torch.cuda.empty_cache()
    mypath = "/home/rshb/myspinner/kidney/data"
    filenames = next(walk(mypath), (None, None, []))[2] 

    # - - - - - - - DATA PARAMETERS - - - - - - - - 
    split = 0.7
    # - - - - - - - HYPER PARAMETERS - - - - - - - - -
    sparsity_threshold = 0.0
    tile_dim = 200
    dense_multiplier = 1
    # - - - - - - - TRAINING PARAMETERS - - - - - - - -
    epochs = 1000
    batch_size = 64

    datagen = Test_Train_Generator(dir_path=mypath,split=split, shuffle=True)

    inference_list = ['NK3']
    train_list = ['NK2']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = IterDataset(generator=datagen.load_data_iter(
        dense_multiplier=dense_multiplier, 
        training=True, 
        tile_dim=tile_dim, 
        sparsity_threshold=sparsity_threshold,
        inference_list=inference_list))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, pin_memory=True)

    test_dataset = IterDataset(generator=datagen.load_data_iter(
        validation=True, 
        tile_dim=tile_dim, 
        sparsity_threshold=sparsity_threshold,
        inference_list=inference_list))  # sparsity_threshold should be 0.0 for inference, 0.0 during validation may throw off LR scheduler
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=1, pin_memory=True)



    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')



    net = UNet()
    # net = torch.nn.DataParallel(net)
    net.to(device=device)

    try:
        train_net(net=net, 
                device=device, 
                train_dataloader=train_loader, 
                test_dataloader=test_loader, 
                n_channels=1,
                n_classes=3,
                epochs=epochs, 
                batch_size=batch_size, 
                save_checkpoint=True, 
                tile_dim=tile_dim, 
                sparsity_threshold=sparsity_threshold, 
                do_validation=False)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise