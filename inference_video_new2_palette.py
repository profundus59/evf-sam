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
from PIL import Image, ImageDraw
import ipdb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    parser.add_argument("--object_id", type=int, default=1)

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
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    image_path = args.image_path
    logging.info("image_path: {}".format(image_path))
    if not os.path.exists(image_path) or not os.path.isdir(image_path): # Simplified check
        print(f"Image folder not found or not a directory: {image_path}")
        exit()

    video_name = os.path.basename(image_path)
    logging.info("video_name: {}".format(video_name))
    
   #TODO: need to make meta_data, which includes prompt and object_id
    meta_data = {"object_id": None, "prompt": None}

    # Get prompt
    prompt = args.prompt
    logging.info("prompt: {}".format(prompt))    
    prompt = [p.strip() for p in prompt.split(",") if p.strip()]
    
    # Get object_id
    
    prompt_num = len(prompts)

    os.makedirs(args.vis_save_path, exist_ok=True)
    tokenizer, model = init_models(args)

    # Get palette
    palette_img = './assets/00000.png'
    palette = Image.open(palette_img).getpalette() # palette is a list of 768 values
    # ipdb.set_trace()

    files = os.listdir(image_path)
    files.sort()

    frame_output_masks = {} # Stores final masks for each frame index
    mask_h, mask_w = 0, 0
    if files:
        try:
            first_img_temp = cv2.imread(os.path.join(image_path, files[0]))
            if first_img_temp is None:
                raise IOError(f"Could not read first image: {os.path.join(image_path, files[0])}")
            mask_h, mask_w = first_img_temp.shape[:2]
            for idx in range(len(files)):
                # Initialize with a zero mask (H, W, 1)
                frame_output_masks[idx] = np.zeros((mask_h, mask_w, 1), dtype=np.float32)
        except Exception as e:
            logger.error(f"Could not determine mask shape or initialize frame_output_masks: {e}")
            return 

    # Main processing loop
    for i, file_name in enumerate(files): # i is the current frame index
        file_path = os.path.join(image_path, file_name)
        try:
            image_np = cv2.imread(file_path)
            if image_np is None:
                logger.warning(f"Could not read image {file_path}. Skipping.")
                continue
            image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.warning(f"Error processing image {file_path}: {e}. Skipping.")
            continue
            
        image_beit = beit3_preprocess(image_np_rgb, args.image_size).to(dtype=model.dtype, device=model.device)

        # Accumulator for all prompt masks for the *current frame i*
        # Ensure it matches the initialized shape, e.g., (mask_h, mask_w, 1)
        current_frame_aggregated_mask = np.zeros((mask_h, mask_w, 1), dtype=np.float32)

        for j in range(prompt_num): # For each prompt
            current_prompt = prompts[j]
            input_ids = tokenizer(current_prompt, return_tensors="pt")["input_ids"].to(device=model.device)

            output_all_frames_propagated = model.inference(
                image_path,  # Used by predictor.init_state to know about the whole video context
                image_beit.unsqueeze(0), # Current frame's visual data
                input_ids, # Current prompt
                i  # current frame index
            )
            # output_all_frames_propagated is like: {frame_k: {obj_id: mask_array_for_frame_k (1,H,W)}}

            # ann_obj_id is 1 as used in evf_sam2_video.py
            ann_obj_id_in_model = 1 #TODO: need to make this dynamic equal to the value of object id in json file.
            
            if i in output_all_frames_propagated and \
               output_all_frames_propagated[i] is not None and \
               ann_obj_id_in_model in output_all_frames_propagated[i]:
                
                mask_for_current_frame_obj = output_all_frames_propagated[i][ann_obj_id_in_model]
                
                if mask_for_current_frame_obj is not None:
                    # mask_for_current_frame_obj is (1, H, W) boolean/float numpy array.
                    # Transpose to (H, W, 1) and ensure it's float for accumulation.
                    # The boolean mask from (logit > 0) needs to be float.
                    processed_mask = mask_for_current_frame_obj.astype(np.float32).transpose(1, 2, 0)

                    # Ensure the mask dimensions match for accumulation
                    if processed_mask.shape[:2] == (mask_h, mask_w):
                         current_frame_aggregated_mask += processed_mask
                    else:
                        # Optional: Resize if model output size is different than expected (e.g. due to internal padding)
                        # logger.warning(f"Mask size mismatch for frame {i}. Expected ({mask_h},{mask_w}), got {processed_mask.shape[:2]}. Resizing.")
                        # resized_mask = cv2.resize(processed_mask, (mask_w, mask_h), interpolation=cv2.INTER_NEAREST)
                        # if resized_mask.ndim == 2: resized_mask = resized_mask[..., np.newaxis]
                        # current_frame_aggregated_mask += resized_mask
                        logger.warning(f"Mask size mismatch for frame {i}. Expected ({mask_h},{mask_w}), got {processed_mask.shape[:2]}. Skipping this mask.")


        # Store the combined mask for frame i. Clip to [0, 1] as masks are binary (0 or 1) and added.
        frame_output_masks[i] = np.clip(current_frame_aggregated_mask, 0, 1)

    # Visualization loop
    for i, file_name in enumerate(files):
        file_path = os.path.join(image_path, file_name)
        img = cv2.imread(file_path)
        if img is None:
            logger.warning(f"Could not read image {file_path} for visualization. Skipping.")
            continue

        # Get the final aggregated and clipped mask for the current frame i
        mask_to_apply = frame_output_masks.get(i) # This is (H,W,1) and float {0,1}

        ipdb.set_trace()
        if mask_to_apply is not None:
            # Ensure mask_to_apply can be broadcast with np.array([0,0,128])
            # img is (H,W,3), mask_to_apply is (H,W,1). Broadcasting should work.
            # Color should be of same dtype as img for addition if img is not float.
            color = np.array([0, 0, 128], dtype=img.dtype) 
            # Ensure mask is also compatible dtype for multiplication, or that img is float
            # If img is uint8, direct multiplication of float mask might require conversion
            if img.dtype == np.uint8:
                 # Overlay: img * (1-alpha) + color * alpha (where alpha is mask)
                 # Simpler: add colored mask where mask is 1
                 overlay = (color * mask_to_apply.astype(img.dtype))
                 out = cv2.addWeighted(img, 1, overlay, 1, 0) # alpha=1 for overlay
            else: # Assuming img is float
                out = img + color * mask_to_apply 
            
            out = np.clip(out, 0, 255).astype(np.uint8) if img.dtype == np.uint8 else out

            save_folder = os.path.join(args.vis_save_path, video_name)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(os.path.join(save_folder, file_name), out)
        else:
            # This case should ideally not be hit if frame_output_masks is initialized for all `i`
            logger.warning(f"No mask found for frame {i} (file {file_name}) during visualization. Saving original.")
            save_folder = os.path.join(args.vis_save_path, video_name)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(os.path.join(save_folder, file_name), img)

if __name__ == "__main__":
    main(sys.argv[1:])