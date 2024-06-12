import argparse
import transformers
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed

from models.modeling_xmodel import XModelForCausalLM
from models.configuration_xmodel import XModelConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="xl", help="")
    parser.add_argument("--model_path", type=str, default="", help="")
    parser.add_argument("--prompt", type=str, default="The capital of China is", help="")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="")
    parser.add_argument("--device", type=str, default="cuda:0", help="")
    args = parser.parse_args()

    model_name, model_path, prompt, max_new_tokens, device = args.model_name, args.model_path, args.prompt, args.max_new_tokens, args.device

    tokenizer = transformers.AutoTokenizer.from_pretrained('models/', use_fast=False, trust_remote_code=True)

    config = XModelConfig.from_name(model_name)
    model = XModelForCausalLM(config)
    PATH = '%s/pytorch_model.bin'%model_path
    model.load_state_dict(torch.load(PATH))
    model.eval()
    model.to(device)
    print('model loaded!')

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    num_tokens = inputs.shape[-1]
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0][num_tokens:]).strip()

    print('prompt: \n', prompt)
    print('response: \n', text)
