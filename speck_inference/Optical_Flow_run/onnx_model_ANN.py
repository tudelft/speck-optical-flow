from pathlib import Path
RUN_DIR = Path(__file__).resolve().parent  # model weights (.pt/.onnx) live here
DATA_DIR = Path(__file__).resolve().parent / "data"  # recorded/generated arrays
DATA_DIR.mkdir(exist_ok=True)

import torch
import torch.nn as nn
import network                    # <- your module with FlowNetwork
                                  #    and WrappedFlowNetwork

# ── instantiate ───────────────────────────────────────────────
model_pt = network.WrappedFlowNetwork(
    encoder_channels=32,
    memory_channels=32,
    decoder_channels=32,
    activation_fn=nn.ReLU,
    final_bias=True,
    padding_mode="reflect",
    scaling=32,
)

# ---- load checkpoint ----
ckpt = torch.load(f"{RUN_DIR}/qs95vlk2_minGRU_depth1_model.pt", map_location="cpu")
# strip “network.” prefix if present
ckpt = {k.replace("network.", ""): v for k, v in ckpt.items()}
missing, unexpected = model_pt.load_state_dict(ckpt, strict=False)
assert not missing and not unexpected, (missing, unexpected)

model_pt.eval()

class ONNXFlowWrapper(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward(self, events, hidden):
        # core still gets the dict it expects
        flow_dict, new_hidden = self.core({"events": events}, hidden)
        # unwrap to plain tensors for ONNX outputs
        return flow_dict["flow"], new_hidden

wrapper = ONNXFlowWrapper(model_pt).cpu()

B, H, W = 1, 128, 128
C_mem = 32         # must match memory_channels
dummy_events = torch.randn(B, 2, H, W)
dummy_hidden = torch.zeros(B, C_mem, H//8, W//8)   # after 3× stride‑2 encoder

flow_out, new_hid = wrapper(dummy_events, dummy_hidden)
print(flow_out.shape, new_hid.shape)   # → (1, 2, 128, 128) (1, 32, 16, 16)

torch.onnx.export(
    wrapper,
    (dummy_events, dummy_hidden),               # positional args
    "qs95vlk2_minGRU_depth1_ANN.onnx",
    input_names=["events", "hidden"],
    output_names=["flow", "new_hidden"],
    dynamic_axes={
        "events": {0: "batch", 2: "height", 3: "width"},
        "hidden": {0: "batch", 2: "h16",    3: "w16"},
        "flow":   {0: "batch", 2: "height", 3: "width"},
        "new_hidden": {0: "batch", 2: "h16", 3: "w16"},
    },
    opset_version=13,
)
print("ONNX export complete → flow_minGRU.onnx")