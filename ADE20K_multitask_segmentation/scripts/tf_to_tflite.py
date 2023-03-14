import os
import pathlib
import datetime
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import torch
import nnio
import torchvision

import seg_training

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, images_directory, imread):
        self.images_filenames = os.listdir(images_directory)
        self.images_directory = images_directory
        self.imread = imread

    def __len__(self):
        return len(self.images_filenames)
    

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = self.imread(os.path.join(self.images_directory, image_filename))
        
        return image


def main(args):
    out_path = args.load_from
    out_name = 'model'

    rep_dataset = SimpleDataset(
        args.dataset_path,
        imread=nnio.Preprocessing(
            resize=(320,320),
            batch_dimension=True,
            divide_by_255=True,
            dtype='float32',
        ),
    )
    
    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(args.load_from)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    def representative_dataset_gen():
        for i in tqdm(range(min(len(rep_dataset), 200))):
            # Get sample input data as a numpy array in a method of your choosing.
            img = rep_dataset[i]
            yield [img]
    converter.representative_dataset = representative_dataset_gen
    print('Quantizing posenet')
    tflite_quant_model = converter.convert()
    out_tflite_path = os.path.join(out_path, f'{out_name}.tflite')
    open(out_tflite_path, "wb").write(tflite_quant_model)

    # Check that output is valid
    torch.manual_seed(0)
    sample_image = torch.randn(1, 3, 320, 224)
    sample_image = sample_image.numpy().transpose([0,2,3,1])
    model = nnio.EdgeTPUModel(out_tflite_path)
    print('EdgeTPU out:')
    print(model(sample_image))

    # Convert to edge TPU
    command = 'edgetpu_compiler -s {} -o {}'.format(
        os.path.join(out_path, f'{out_name}.tflite'),
        out_path,
    )
    print('Executing command:')
    print(command)
    os.system(command)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert saved model to onnx.')

    parser.add_argument('--load_from', type=str,
                        help='path to tensorflow .pb model')
    parser.add_argument('--dataset_path', type=str,
                        help='path to pascal voc dataset')

    args = parser.parse_args()

    main(args)
