from PIL import Image
import os
from tqdm import tqdm
import json
import ipdb
import subprocess


def get_folders_from_path(path):
    # extract part from folder name after 'val/'
    folders = [f.path.split('val/')[1] for f in os.scandir(path) if f.is_dir() and 'val/' in f.path]
    folders.sort(key=lambda x: int(x.split('_')[0]))
    
    return folders


def extract_info(vid_name, path2, img_folder_path):
    with open(path2, "r") as f:
        task = json.load(f)

    video_titles = list(task['videos'].keys())
    expression_list = []

    for i in range(len(video_titles)):
        if video_titles[i] == vid_name:
            expressions = task['videos'][video_titles[i]]['expressions']
            expression_size = len(expressions)
            for j in range(expression_size):
                expression = task['videos'][video_titles[i]]['expressions'][str(j)]['exp']
                expression_list.append(expression)
                # print(expression)

    vid_path = os.path.join(img_folder_path, vid_name)    
    
    return expression_list, vid_path, expression_size


if __name__ == "__main__":
    vid_folder_path = './assets/dataset_visor/JPEGImages_Sparse/val'
    prompt_path = './assets/dataset_visor/ImageSets/val_meta_expressions_promptaction.json'
    
    bash_script = './jh_infer_visor.sh'

    vid_subfolders = get_folders_from_path(vid_folder_path)
    size = len(vid_subfolders)
    print(f"Number of subfolders: {size}")

    # ipdb.set_trace()
    for i in range(size):
        expression_list, subvid_path, expression_size = extract_info(vid_subfolders[i], prompt_path, vid_folder_path)
        print(f"Processing {vid_subfolders[i]} with {expression_size} expressions.")
        
        # ipdb.set_trace()
        command = [
            bash_script,
            subvid_path,
            str(expression_list), # take all the expressions from whole video
        ]
        # ipdb.set_trace()
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            # ipdb.set_trace()
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            print(f"Stdout:\n{e.stdout}")
            print(f"Stderr:\n{e.stderr}")
            raise
        
        ipdb.set_trace()
    print(f"Bash script executed successfully.")