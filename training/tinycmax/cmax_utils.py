import torch
from torch.nn.utils.rnn import pad_sequence


def extract_events_from_frames(frames):
    """
    Extract 'events' from frames whose pixels have channels
    (neg polarity count, pos polarity count, avg quantized ts).
    """
    # get shape
    b, _, _, _, _ = frames.shape

    # get indices of nonzero counts directly (wo creating mask)
    nonzero_indices = frames[:, :2].nonzero(as_tuple=True)
    bi, pi, zi, yi, xi = nonzero_indices  # b, p, d, h, w

    # combine nonzero polarities with xy and ts coordinates
    # start from 2: so add 2 to pi
    avg_ts = frames[bi, 2 + pi, zi, yi, xi] + zi  # increment by passes
    xyz = torch.stack([xi, yi, avg_ts, zi], dim=1)
    pol = frames[bi, pi, zi, yi, xi] * (2 * pi - 1)
    combined = torch.cat([xyz, pol.unsqueeze(1)], dim=1)

    # get count per batch dim
    # split into list per batch element
    counts = torch.bincount(bi, minlength=b)
    outputs = combined.split(counts.tolist())

    # pad to batch
    output = pad_sequence(outputs, batch_first=True)

    return output


def format_events(events, counts, stack=False):
    """
    Go from list of padded (b, n, 4) events with (t, y, x, p)
    to padded (b, n, 5) events with (x, y, t, pass, p)
    or (b, n, d, 5) if stack is true.
    """
    max_counts = [c.max() for c in counts]  # per batch
    if stack:
        max_counts = [max(max_counts)] * len(max_counts)  # overall
    output = []
    for i, (ev, c) in enumerate(zip(events, max_counts)):
        t, y, x, p = ev[:, :c].unbind(-1)
        z = torch.ones_like(t) * i
        output.append(torch.stack([x, y, t + i, z, p], dim=-1))
    output = torch.cat(output, dim=1) if not stack else torch.stack(output, dim=2)
    return output
