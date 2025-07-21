import torch
import torch.nn as nn
import network                               # your module

# 1. Build the original core model
core = network.WrappedFlowNetwork(
    encoder_channels=32,
    memory_channels=32,
    decoder_channels=32,
    activation_fn=nn.ReLU,
    final_bias=True,
    padding_mode="reflect",
    scaling=32,
)

# 2. Load checkpoint (strip the 'network.' prefix)
state = torch.load(
    "Optical_Flow_tinycmax/qs95vlk2_minGRU_depth1_model.pt",
    map_location="cpu",
)
core.load_state_dict({k.replace("network.", ""): v for k, v in state.items()})
core.eval()

# 3. **Thin wrapper**: converts 2‑tensor input → dict the core expects
class ONNXFlowWrapper(nn.Module):
    def __init__(self, core_model):
        super().__init__()
        self.core = core_model

    def forward(self, events, hidden):
        return self.core({"events": events}, hidden)

wrapper = ONNXFlowWrapper(core).cpu()      # or .to(device)

# 4. Dummy inputs
B, C, H, W = 1, 32, 16, 16          # hidden shape
dummy_events = torch.randn(B, 2, 128, 128)
dummy_hidden = torch.zeros(B, C, H, W)

# **Test once in eager mode**
flow_out, new_hidden = wrapper(dummy_events, dummy_hidden)
#print(flow_out.shape, new_hidden.shape)    # should be (1,2,128,128) and (1,32,16,16)

# 5. Export → ONNX
torch.onnx.export(
    wrapper,
    (dummy_events, dummy_hidden),           # ← still two tensors
    "Optical_Flow_tinycmax/qs95vlk2_minGRU_depth1_ANN.onnx",
    input_names=["events", "hidden"],
    output_names=["flow", "new_hidden"],
    dynamic_axes={
        "events": {0: "batch", 2: "height", 3: "width"},
        "hidden": {0: "batch", 2: "height", 3: "width"},
        "flow": {0: "batch", 2: "height", 3: "width"},
        "new_hidden": {0: "batch", 2: "height", 3: "width"},
    },
    opset_version=13,
)

print("ONNX export complete.")
