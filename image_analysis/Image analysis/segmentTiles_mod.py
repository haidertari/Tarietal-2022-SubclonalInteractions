import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from ResNetUNet import ResNetUNet
from torchvision import transforms

def segmentTiles(tile_path, classifier_path, out_path, tile_in_size=[224, 224, 3], ignore_border=10, tile_gap=[112, 112], batch_size=30, threshold=0.5, gpu=True, overwrite=True):
    def process_tiles(tiles, tile_positions, mask, counts):
        output = F.sigmoid(model(tiles)).data

        if gpu:
            output = output.cpu()

        output = np.squeeze(output)
        #print(mask.shape)
        for i in range(output.shape[0]):
            mask[(tile_positions[i][0]+ignore_border):(tile_positions[i][2]-ignore_border), (tile_positions[i][1]+ignore_border):(tile_positions[i][3]-ignore_border)] += output[i, ignore_border:(tile_positions[i][2]-tile_positions[i][0]-ignore_border), ignore_border:(tile_positions[i][3]-tile_positions[i][1]-ignore_border)].numpy()
            counts[(tile_positions[i][0]+ignore_border):(tile_positions[i][2]-ignore_border), (tile_positions[i][1]+ignore_border):(tile_positions[i][3]-ignore_border)] += 1



    model = torch.load(classifier_path, map_location=lambda storage, loc: storage)

    if gpu:
        model = model.cuda().eval()
    else:
        model = model.cpu().eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
    ])

    os.makedirs(out_path, exist_ok=True)

    image_paths = glob.glob(os.path.join(tile_path, '*.tif'))

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR);
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        counts = np.zeros(image.shape[:2], dtype=np.float32)

        tiles = torch.tensor(np.zeros((batch_size, tile_in_size[2], tile_in_size[0], tile_in_size[1])), dtype=torch.float)
        tile_positions = []
        tile_idx = 0

        if gpu:
            tiles = tiles.cuda()
        else:
            tiles = tiles.cpu()

        print("Segmenting: "+os.path.basename(image_path))

        for x in range(0, image.shape[0], tile_gap[0]):
            for y in range(0, image.shape[1], tile_gap[1]):
                tile_width = min(tile_in_size[0], image.shape[0]-x)
                tile_height = min(tile_in_size[1], image.shape[1]-y)

                tile_image = np.zeros(tile_in_size,dtype=np.uint8)

                tile_image[:tile_width, :tile_height, :] = image[x:(x+tile_width), y:(y+tile_height), :]
                tile_positions = tile_positions + [[x, y, x+tile_width, y+tile_height]]

                tiles[tile_idx, :, :, :] = transform(tile_image)
                #print(np.max(tiles.cpu().numpy()))

                tile_idx += 1

                if tile_idx == tiles.size()[0]:
                    process_tiles(tiles, tile_positions, mask, counts)
                    tile_idx = 0
                    tile_positions = []

        process_tiles(tiles[:tile_idx, :, :, :], tile_positions, mask, counts)

        mask = mask/counts;


        cv2.imwrite(os.path.join(out_path, os.path.basename(image_path)), (255*mask).astype(np.uint8))
