import torch
import torch.nn.functional as F
from tqdm import trange

class CombinedDecoder(torch.nn.Module):
    """ Creation of a class to combine the decoder and the lm head """
    def __init__(self, decoder, lm_head, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.config = config
    def forward(self, input_ids, encoder_hidden_states):
        decoder_output = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)[0] * \
                         (self.config.d_model ** -0.5)
        return self.lm_head(decoder_output)

class SimplifiedT5Encoder(torch.nn.Module):
    """ Creation of a class to output only the last hidden state from the encoder """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, *input, **kwargs):
        return self.encoder(*input, **kwargs)[0]

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

class GenerativePagasus(torch.nn.Module):
    def __init__(self, encoder, decoder_with_lm_head, tokenizer, onnx=False, cuda=False):
        super().__init__()
        self.encoder = encoder
        self.decoder_with_lm_head = decoder_with_lm_head
        self.tokenizer = tokenizer
        self.onnx = onnx
        self.cuda = cuda

    def forward(self, prompt, max_length, temperature=1., repetition_penalty=1., top_k=50, top_p=0, max_context_length=512):
        with torch.no_grad():
            new_tokens = torch.tensor(())
            new_logits = []
            generated = torch.tensor(self.tokenizer(prompt)['input_ids'])[:max_context_length - 1].unsqueeze(0)
            if self.cuda and not self.onnx:
                generated = generated.cuda()

            # Getting encoder past
            if self.onnx:
                encoder_outputs_prompt = self.encoder.run(None, {"input_ids": generated.cpu().numpy()})[0]
            else:
                encoder_outputs_prompt = self.encoder(generated)

            # The sequence now needs to start with a
            generated = torch.zeros((1,1), dtype=torch.long)
            if self.cuda and not self.onnx:
                generated = generated.cuda()

            for _ in trange(max_length):
                if self.onnx:
                    outputs = torch.tensor(self.decoder_with_lm_head.run(None, {"input_ids": generated.cpu().numpy(),
                                                   "encoder_hidden_states": encoder_outputs_prompt})[0][0])
                else:
                    outputs = self.decoder_with_lm_head(input_ids=generated,
                                                        encoder_hidden_states=encoder_outputs_prompt)[0]
                next_token_logits = outputs[-1, :] / (temperature if temperature > 0 else 1.0)
                if int(next_token_logits.argmax()) == 1:
                    break
                new_logits.append(next_token_logits)
                for _ in set(generated.view(-1).tolist()):
                    next_token_logits[_] /= repetition_penalty
                if temperature == 0:  # greedy sampling:
                    next_token = torch.argmax(next_token_logits).unsqueeze(0)
                else:
                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                new_tokens = torch.cat((new_tokens, next_token), 0)
            new_tokens = new_tokens.to(torch.int64)
            return self.tokenizer.decode(new_tokens), new_logits