import torch.nn as nn

#from tinycmax.blocks_original import conv_encoder, LazyConvGru, upsample_decoder
from blocks import conv_encoder, LazyConvGru, upsample_decoder
from network_utils import NetworkWrapper





class FlowNetwork(nn.Module):
    """
    Optical flow prediction network following IDNet (Wu et al., ICRA'24).
    """

    mode = "flow"

    def __init__(
        self,
        memory_channels,
        decoder_channels,
        activation_fn,
        final_bias,
        padding_mode,
        scaling,
    ):
        super().__init__()

        self.scaling = scaling

        self.memory = LazyConvGru(memory_channels, 3, padding_mode=padding_mode)
        self.decoder = upsample_decoder(
            decoder_channels, activation_fn, final_bias, padding_mode=padding_mode, mode=self.mode
        )

    def forward(self, input, hidden=None):
        frame = input["events"]  # .events incompatible with torch.compile?
        memory = self.memory(frame, hidden)
        flow_map = self.decoder(memory)

        flow_map *= self.scaling

        return dict(flow=flow_map), memory


class WrappedFlowNetwork(NetworkWrapper, FlowNetwork):
    def forward(self, input, hidden=None):
        output, hidden = FlowNetwork.forward(self, input, hidden)
        self.set_state(hidden)
        return output, hidden
