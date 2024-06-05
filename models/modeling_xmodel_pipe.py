import torch
from deepspeed.pipe import LayerSpec, PipelineModule
from models.modeling_xmodel import XModelForCausalLM, Model, RMSNorm, DecoderLayer
# from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional, List, Tuple, Union
from transformers.utils import logging
from megatron import get_args

# from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# from ..models.configuration_xmodel import XModelConfig

logger = logging.get_logger(__name__)
torch2 = torch.__version__.split('.')[0] == '2'


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device
        )
        # print('expanded_attn_mask',expanded_attn_mask.shape,expanded_attn_mask)
        # print('combined_attention_mask',combined_attention_mask.shape,combined_attention_mask)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                                                                       combined_attention_mask
        )

    return combined_attention_mask


class EmbeddingPipeLayer(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.embed_tokens = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx)
        self.config = config
        self.gradient_checkpointing = False

    def forward(self, input_data, **kwargs):  # -> PipeDecoderLayerInputOutput:
        # torch.set_grad_enabled(True)
        input_ids = input_data[0]
        position_ids = input_data[1]
        attention_mask = None  # = data.attention_mask
        # position_ids = None  # = data.position_ids
        past_key_values = None  # = data.past_key_values
        inputs_embeds = None  # = data.inputs_embeds
        use_cache = None  # = data.use_cache
        output_attentions = None  # = data.output_attentions
        output_hidden_states = None  # = data.output_hidden_states
        return_dict = None  # = data.return_dict

        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        # use_cache = use_cache if use_cache is not None else self.config.use_cache
        #
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # # embed positions
        # if attention_mask is None:
        #     attention_mask = torch.ones(
        #         (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        #     )
        # attention_mask = _prepare_decoder_attention_mask(
        #     attention_mask, (batch_size,
        #                      seq_length), inputs_embeds, past_key_values_length
        # )

        hidden_states = inputs_embeds

        # debug 2024-01-25 hxc deepspeed_pipe_engine all pf32 input require grad
        # attention_mask.requires_grad=True
        # attention_mask = attention_mask.to(dtype=torch.int64) # int type is executable but loss calculation is corrupted
        # debug 2024-01-30 hxc deepspeed_pipe_engine pp:check requires_grad,tp:forbid requires_grad, so attention_mask be ignored
        res = (hidden_states, position_ids)
        # res = (hidden_states)
        return res


class DecoderPipeLayer(torch.nn.Module):
    def __init__(self, config, layer_index) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.decoder_layer = DecoderLayer(config=config)
        # if not hasattr(self, '_args'):
        self._args = get_args()

    def forward(self, args, **kwargs):  # -> PipeDecoderLayerInputOutput:
        # if len(args) == 1:
        #     args = args
        # hidden_states, attention_mask, position_ids = args[0],args[1],args[2]
        hidden_states, position_ids = args[0], args[1]
        # hidden_states= args[0]

        # attention_mask = self._args.attn_mask
        batch_size, seq_length, _ = hidden_states.shape
        past_key_values_length=0
        attention_mask=None
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool)
        attention_mask = _prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length)

        cur_device = next(self.decoder_layer.parameters()).device
        # print('cur_device',cur_device)

        layer_outputs = self.decoder_layer(
            hidden_states=hidden_states.to(cur_device),
            attention_mask=attention_mask.to(cur_device),
            position_ids=position_ids.to(cur_device),
            past_key_value=None,  # past_key_value,
            output_attentions=None,
            use_cache=False,
        )
        hidden_states = layer_outputs[0]

        # res = (hidden_states, attention_mask, position_ids)
        res = (hidden_states, position_ids)
        # res = (hidden_states)
        return res


class LayerNormPipeLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, inputs):
        # torch.set_grad_enabled(True)
        hidden_states, *_ = inputs
        last_hidden_states = self.norm(hidden_states)

        return last_hidden_states


class LMHeadPipeLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, inputs):
        # torch.set_grad_enabled(True)
        logits = self.lm_head(inputs)

        return logits


def loss_fn(outputs, labels):
    # torch.set_grad_enabled(True)
    logits = outputs
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

    return loss


def get_layers_from_config(model_config):
    layers = [
        LayerSpec(EmbeddingPipeLayer, model_config),
        *[LayerSpec(DecoderPipeLayer, model_config, idx) for idx in range(model_config.num_hidden_layers)],
        LayerSpec(LayerNormPipeLayer, model_config),
        LayerSpec(LMHeadPipeLayer, model_config)
    ]
    return layers
