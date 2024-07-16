import torch
import struct
import os

# Constants
VERSION = 1
MAGIC_NUMBER = 0xDEADBEEF

def write_string(f, s):
    encoded = s.encode('utf-8') + b'\0'
    f.write(encoded)

def export_pytorch_checkpoint(checkpoint_path, config_path, output_path):
    model = torch.load(checkpoint_path)

    with open(config_path, "r") as f:
        config = f.read()

    with open(output_path, "wb") as f:
        # Write header
        f.write(struct.pack("<II", VERSION, MAGIC_NUMBER))

        # Write config
        write_string(f, config)

        # Align to 256 bytes
        current_offset = f.tell()
        padding = (256 - current_offset % 256) % 256
        f.write(b'\0' * padding)

        # Prepare to store tensor information
        tensor_infos = []
        current_offset = f.tell()

        # Write tensors
        for name, tensor in model.items():
            # Align to 256 bytes
            padding = (256 - current_offset % 256) % 256
            f.write(b'\0' * padding)
            current_offset += padding

            # Write tensor data
            tensor_bytes = tensor.detach().cpu().to(torch.float32).numpy().tobytes()
            f.write(tensor_bytes)

            # Store tensor information
            tensor_infos.append({
                "name": name,
                "offset": current_offset,
                "size": len(tensor_bytes),
                "shape": tensor.shape,
                "encoding": "float32",
            })
            current_offset += len(tensor_bytes)

        # Write tensor information
        footer_start = current_offset
        for info in tensor_infos:
            write_string(f, info["name"])
            f.write(struct.pack("<QQ", info["offset"], info["size"]))
            f.write(struct.pack("<I", len(info["shape"])))
            f.write(struct.pack(f"<{len(info['shape'])}I", *info["shape"]))
            write_string(f, info["encoding"])

        # Write footer metadata
        print(footer_start, len(tensor_infos))
        f.write(struct.pack("<QQ", footer_start, len(tensor_infos)))
        print(f.tell())

checkpoint_path = "Meta-Llama-3-8B/consolidated.00.pth"
config_path = "Meta-Llama-3-8B/params.json"
output_path = "output_checkpoint.bin"
export_pytorch_checkpoint(checkpoint_path, config_path, output_path)
