import numpy as np
import os
import torch
import ezdxf
from transformers import AutoTokenizer
from dxf_preprocessor import convert_to_primitives, primitives_to_tensor

class DXFDataset:
    def __init__(self, tensor_folders_dir):
        self.tensor_folders_dir = tensor_folders_dir
        self.folder_names = os.listdir(tensor_folders_dir)

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):
        folder_name = self.folder_names[idx]
        folder_path = os.path.join(self.tensor_folders_dir, folder_name)
        arg_tensor = torch.load(os.path.join(folder_path, "arg_tensor.pt"))
        arg_types_tensor = torch.load(os.path.join(folder_path, "arg_types_tensor.pt"))
        prim_types_tensor = torch.load(os.path.join(folder_path, "prim_types_tensor.pt"))
        text_input_ids = torch.load(os.path.join(folder_path, "text_input_ids.pt"))
        text_attention_mask = torch.load(os.path.join(folder_path, "text_attention_mask.pt"))
        tensor_data = {
            "arg_tensor": arg_tensor,
            "arg_types_tensor": arg_types_tensor,
            "prim_types_tensor": prim_types_tensor,
            "text_input_ids": text_input_ids,
            "text_attention_mask": text_attention_mask
        }
        return tensor_data

# save output in a folder: out_dir/dxf_name/(tensors)
# tensors are: 
def save_dxf_to_tensor(dxf_path, out_dir, tok, max_length=64):
    dxf_file = ezdxf.readfile(dxf_path)
    primitives, primitives_text = convert_to_primitives(dxf_file)
    arg_tensor, arg_types_tensor, prim_types_tensor = primitives_to_tensor(primitives, arg_vec_length=16, num_arg_types=10, num_prim_types=10)

    texts = [text.plain_text() for text in primitives_text]
    if texts:
        tokenized = tok(texts, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt") 
        text_input_ids = tokenized["input_ids"]
        text_attention_mask = tokenized["attention_mask"]  
    # save
    out_folder = os.path.join(out_dir, os.path.basename(dxf_path).split(".")[0])
    os.makedirs(out_folder, exist_ok=True)
    torch.save(arg_tensor, os.path.join(out_folder, "arg_tensor.pt"))
    torch.save(arg_types_tensor, os.path.join(out_folder, "arg_types_tensor.pt"))
    torch.save(prim_types_tensor, os.path.join(out_folder, "prim_types_tensor.pt"))
    if texts:
        torch.save(text_input_ids, os.path.join(out_folder, "text_input_ids.pt"))
        torch.save(text_attention_mask, os.path.join(out_folder, "text_attention_mask.pt"))


if __name__ == "__main__":
    ## precompute tensors from dxf files
    # tunable
    # need to recompute, if you change tokenizer or max_length
    dxf_dir = "/srv/scratch/weixuan2/datasets/dxf_raw"
    out_dir = "/srv/scratch/weixuan2/datasets/dxf_converted"
    model_name = "google/byt5-small"
    max_length = 64

    # all dxf file paths
    dxf_names = os.listdir(dxf_dir)
    dxf_paths = [os.path.join(dxf_dir, name) for name in dxf_names if name.endswith(".dxf")]
    # text tokenizer (no trainining required, so safe to save)
    tok = AutoTokenizer.from_pretrained(model_name)
    for dxf_name, dxf_path in zip(dxf_names, dxf_paths):
        try:
            save_dxf_to_tensor(dxf_path, out_dir, tok, max_length=max_length)
            print("Processed ", dxf_name)
        except:
            print("Failed to process ", dxf_name)
    
