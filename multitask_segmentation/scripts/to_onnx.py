'''
Convert saved model to onnx
'''

import os
import pathlib
import datetime
import argparse
import numpy as np
import torch
import torch.onnx
import onnx
import onnxruntime
from shutil import copyfile

import seg_training


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def main(config):
    # Default name
    config.out_name = config.out_name or f'segmentation'

    # Create lit module
    if config.load_from is not None:
        lit_module = seg_training.pl_module.LitSeg.load_from_checkpoint(checkpoint_path=config.load_from, strict=False)
    else:
        print('\n\nWARNING!!!!\nArgument --load_from is not set!!!!!!\n\n')
        lit_module = seg_training.pl_module.LitSeg(**config.__dict__)
    lit_module.eval()

    # Create path
    now = datetime.datetime.now()
    out_path = os.path.join(args.out_path, f'{now.strftime("%Y.%m.%d")}_{config.out_name}')
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)

    # Copy the config file
    copyfile(config.config, os.path.join(out_path, 'config.yaml'))

    # Convert to openvino script
    openvino_command = f'''
docker run --rm -it \\
    -v /etc/timezone:/etc/timezone:ro \\
    -v /etc/localtime:/etc/localtime:ro \\
    -v {pathlib.Path(out_path).absolute()}:/input \\
    openvino/ubuntu18_dev \\
    python3 deployment_tools/model_optimizer/mo.py \\
    --input_model /input/{config.out_name}_op{config.opset}.onnx \\
    --model_name {config.out_name}_fp16 \\
    --data_type FP16 \\
    --output_dir /input/ \\
    --input_shape "[1,3,{config.image_h},{config.image_w}]"
'''

    # Create readme file
    readme_path = os.path.join(out_path, 'readme.md')
    with open(readme_path, 'w') as readme:
        readme.write(f'# Segmentation ONNX model\n\n')
        readme.write(f'Created at {now.strftime("%H:%M:%S %d.%m.%Y")}\n\n')
        nnio_preproc = f'''
preprocessing = nnio.Preprocessing(
    resize=({config.image_w}, {config.image_h}),
    channels_first=True,
    divide_by_255=False,
    dtype='float32',
    batch_dimension=True
)
'''
        readme.write(f'## For nnio:\n```{nnio_preproc}```\n')
        nnio_preproc_tflite = nnio_preproc.replace('channels_first=True', 'channels_first=False')
        readme.write(f'or for tflite:\n```{nnio_preproc_tflite}```\n')
        readme.write(f'\n## To convert to OpenVINO:\n```{openvino_command}```')

    # Convert model to onnx
    torch.manual_seed(0)
    sample_image = torch.randn(
        1, 3, config.image_h, config.image_w, requires_grad=True)
    torch_out = lit_module(sample_image)
    print('torch_out.shape:', to_numpy(torch_out).shape)
    onnx_model_path = os.path.join(out_path, f'{config.out_name}_op{config.opset}.onnx')
    print('Exporting model as:', onnx_model_path)
    lit_module.to_onnx(
        onnx_model_path,
        sample_image,
        export_params=True,
        opset_version=config.opset,
        do_constant_folding=True,
        input_names = ['image'],
        output_names = ['mask'],
    )

    # Check model
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # Check runtime
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    # compute ONNX Runtime output prediction
    ort_inputs = {
        'image': to_numpy(sample_image),
    }
    ort_outs = ort_session.run(None, ort_inputs)
    print(np.unique(ort_outs))
    print(np.unique(to_numpy(torch_out)))
    if np.allclose(ort_outs[0], to_numpy(torch_out), rtol=1e-3, atol=1e-3):
        print('Model is working correctly')
    else:
        print(ort_outs[0])
        raise BaseException('ONNX model gave different results from torch model''s')

    print('\nTo convert model to openVINO, please run:\n', openvino_command)
    print('Success!')
    print('\nAlso check out', readme_path)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert saved model to onnx.')

    parser.add_argument('--config', type=str,
                        help='configuration file in yaml format (ex.: configs/config_warehouse.yaml)')
    parser.add_argument('--load_from', type=str,
                        default=None,
                        help='trained model saved using pytorch-lightning')
    parser.add_argument('--out_path', type=str,
                        default='./onnx_output',
                        help='path to the output folder')
    parser.add_argument('--out_name', type=str,
                        default=None,
                        help='name of the output .onnx file')
    parser.add_argument('--opset', type=int,
                        default=11,
                        help='ONNX opset version')

    parser = seg_training.pl_module.LitSeg.add_argparse_args(parser)
    parser = seg_training.datasets.pl_datamodule.SegDataModule.add_argparse_args(parser)

    parser = seg_training.utils.initialization.set_argparse_defaults_from_yaml(parser)

    args = parser.parse_args()

    main(args)
