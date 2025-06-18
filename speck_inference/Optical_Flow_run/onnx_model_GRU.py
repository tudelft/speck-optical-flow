import network_woenc_GRU 
import torch
import torch.nn as nn

model = network_woenc_GRU.WrappedFlowNetwork(
    memory_channels=64,
    decoder_channels=64,
    activation_fn=nn.ReLU,
    final_bias=True,
    padding_mode="reflect",
    scaling=32
)

state_dict = torch.load("Optical_Flow_tinycmax/rqe_iterative_model_hand.pt", map_location="cuda" if torch.cuda.is_available() else "cpu")
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

B, C, H, W = 1, 64, 16, 16  # Adjust to match your expected event shape
dummy_events = torch.randn(B, C, H, W).to(device)
dummy_hidden = torch.zeros(B, C, H, W).to(device) 

# Wrap input in dict, as your model expects
#input_dict = {"events": dummy_events}

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_events, dummy_hidden),  # Tuple of args
    "Optical_Flow_tinycmax/mem_decoder_network_64.onnx",
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


print("ONNX export complete: mem_decoder_network_64.onnx")