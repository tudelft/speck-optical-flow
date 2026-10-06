from pathlib import Path
RUN_DIR = Path(__file__).resolve().parent  # model weights (.pt/.onnx) live here
DATA_DIR = Path(__file__).resolve().parent / "data"  # recorded/generated arrays
DATA_DIR.mkdir(exist_ok=True)

import network_woenc_minGRU 
import torch
import torch.nn as nn

model = network_woenc_minGRU.WrappedFlowNetwork(
    memory_channels=32,
    decoder_channels=32,
    activation_fn=nn.ReLU,
    final_bias=True,
    padding_mode="reflect",
    scaling=32
)

state_dict = torch.load(f"{RUN_DIR}/btlez3ib_model.pt", map_location="cuda" if torch.cuda.is_available() else "cpu")
print(state_dict.keys())

## Skipping the encoder states
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("network.", "")  # strip the prefix
    if new_key.startswith("encoder."):
        continue 
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(model)

B, C, H, W = 1, 32, 16, 16  # Adjust to match your expected event shape
dummy_events = torch.randn(B, C, H, W).to(device)
dummy_hidden = torch.zeros(B, C, H, W).to(device) 

# Wrap input in dict, as your model expects
#input_dict = {"events": dummy_events}

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_events, dummy_hidden),  # Tuple of args
    f"{RUN_DIR}/mem_decoder_network.onnx",
    input_names=["events", "hidden"],
    output_names=["flow", "new_hidden"],
    dynamic_axes={
        "events": {0: "batch", 2: "height", 3: "width"},
        "hidden": {0: "batch", 2: "height", 3: "width"},
        "flow": {0: "batch", 2: "height", 3: "width"},
        "new_hidden": {0: "batch", 2: "height", 3: "width"},
    },
    opset_version=13
)


print("ONNX export complete: wrapped_flow_network.onnx")