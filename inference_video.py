import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
from model.segment_anything.utils.transforms import ResizeLongestSide
import ipdb

def parse_args(args):
    parser = argparse.ArgumentParser(description="EVF infer")
    parser.add_argument("--version", required=True)
    parser.add_argument("--vis_save_path", default="./outputs/infer_video", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=224, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--model_type", default="ori", choices=["ori", "effi", "sam2"])
    parser.add_argument("--image_path", type=str, default="assets/zebra.jpg")
    parser.add_argument("--prompt", type=str, default="zebra top left")
    
    return parser.parse_args(args)

def beit3_preprocess(x: np.ndarray, img_size=224) -> torch.Tensor:
    '''
    preprocess for BEIT-3 model.
    input: ndarray
    output: torch.Tensor
    '''
    beit_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC), 
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return beit_preprocess(x)

def init_models(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        padding_side="right",
        use_fast=False,
    )
    # ipdb.set_trace()
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    if args.model_type=="sam2":
        from model.evf_sam2_video import EvfSam2Model
        model = EvfSam2Model.from_pretrained(
            args.version, low_cpu_mem_usage=True, **kwargs
        )

    if (not args.load_in_4bit) and (not args.load_in_8bit):
        model = model.cuda()
    model.eval()

    return tokenizer, model

def main(args):
    args = parse_args(args)
    # use float16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # clarify IO
    image_path = args.image_path # path of folder containing images
    if not os.path.exists(image_path):
        print("File not found in {}".format(image_path))
        exit()
    if not isinstance(image_path, str) and os.path.isdir(image_path):
        print("Path has to be a folder path, check again.")
        exit()
    # ipdb.set_trace()
    # Extract video name from image path
    video_name = os.path.basename(image_path)
    
    prompt = args.prompt
    # ipdb.set_trace()
    prompt = [p.strip() for p in prompt.split(",") if p.strip()]
    prompt_num = len(prompt)

    os.makedirs(args.vis_save_path, exist_ok=True)

    # initialize model and tokenizer
    tokenizer, model = init_models(args)

    # preprocess
    files = os.listdir(image_path) #files: image files
    files.sort()
    # ipdb.set_trace()

    # Total output mask collection:
    frame_output = {}
    for i in range(len(files)):
        frame_output[i] = None       

    for i, file in enumerate(files):
        file_path = os.path.join(image_path, file)

        image_np = cv2.imread(file_path) 
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_beit = beit3_preprocess(image_np, args.image_size).to(dtype=model.dtype, device=model.device) #shape: (3, 224, 224)
        # ipdb.set_trace()

        # for each expression prompt in the video
        for j in range(prompt_num): 
            input_ids = tokenizer(prompt[j], return_tensors="pt")["input_ids"].to(device=model.device)
            # ipdb.set_trace()

            # infer
            output = model.inference(
                image_path, 
                image_beit.unsqueeze(0),
                input_ids,
                # original_size_list=original_size_list,
            )
            # output shape: 
            # output.get(frame numbers)[1].shape: (1, 480, 854)

            for k in output.keys():
                output[k][1] = output[k][1].transpose(1, 2, 0)

            frame_output_temp = {}
            # ipdb.set_trace()

            # Update frame_output for each frame in video, for current expression.
            for k, val in output.items():
                if frame_output.get(k) is None:
                    frame_output[k] = np.zeros_like(val[1])
                frame_output_temp[k] = val[1]
                frame_output[k] = frame_output.get(k) + frame_output_temp.get(k) # update frame_output

            # ipdb.set_trace()

    for i, file in enumerate(files):
        # save visualization
        files = os.listdir(image_path)
        files.sort()
        img = cv2.imread(os.path.join(image_path, file)) #shape: (480, 854, 3)
        # ipdb.set_trace()
        out = img + np.array([0,0,128]) * frame_output.get(i)[1]
        # make directory for current video inside args.vis_save_path
        os.makedirs(os.path.join(args.vis_save_path, video_name), exist_ok=True)
        cv2.imwrite(os.path.join(args.vis_save_path, video_name, file), out)

if __name__ == "__main__":
    main(sys.argv[1:])