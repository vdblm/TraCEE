import torch
import torch.nn as nn

from transformers import GPT2Model, GPT2Config
from src.configs import TraCEEConfig


def build_model(conf: TraCEEConfig):
    model = TransformerModel(
        n_dims=conf.model.n_dims,
        n_positions=conf.model.n_positions,
        n_embd=conf.model.n_embd,
        n_layer=conf.model.n_layer,
        n_head=conf.model.n_head,
    )
    return model


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 2)  # output mean, log-var

    def forward(self, xtys):
        xtys_dim = xtys.shape[-1]
        if xtys_dim <= self.n_dims:
            # pad last dim with zeros
            xtys = nn.functional.pad(xtys, (0, self.n_dims - xtys_dim))
        else:
            raise ValueError(
                f"Input dimension {xtys_dim} greater than model dimension {self.n_dims}"
            )

        embeds = self._read_in(xtys)
        # shape: (batch_size, n_positions, n_embd)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        # shape: (batch_size, n_positions, 2)
        prediction = self._read_out(output)
        return prediction[:, :, 0], prediction[:, :, 1]
