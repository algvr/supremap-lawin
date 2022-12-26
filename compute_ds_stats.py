import argparse
import os
import numpy as np
from PIL import Image


IMAGE_EXTS = ['.png', '.jpg', '.jpeg']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', action='append', help='Directory or list of directories to search for images', type=str)
    args = parser.parse_args()
    img_means = []
    img_stds = []

    for dir_path in args.dir:
        for root, dirs, files in os.walk(dir_path):
            for filename in files:
                if not any(filter(lambda ext: filename.lower().endswith(ext), IMAGE_EXTS)):
                    continue
                
                file_path = os.path.join(root, filename)
                with Image.open(file_path) as img:
                    arr = np.array(img)
                    num_channels = arr.shape[-1]
                    for channel_idx in range(num_channels):
                        mean = arr[:, :, channel_idx].mean()
                        std = arr[:, :, channel_idx].std()
                        if len(img_means) <= channel_idx:
                            img_means.append([])
                            img_stds.append([])
                        img_means[channel_idx].append(mean)
                        img_stds[channel_idx].append(std)
                    
    
    print('Means: ' + ', '.join(['%.4f' % np.mean(m) for m in img_means]))
    print('StDevs: ' + ', '.join(['%.4f' % np.mean(s) for s in img_stds]))
