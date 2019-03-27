from pytorch_pretrained_bert import TransfoXLTokenizer, TransfoXLModel, TransfoXLLMHeadModel, TransfoXLCorpus, TransfoXLConfig
from pytorch_pretrained_bert.tokenization_transfo_xl import get_lm_corpus


import torch
import numpy as np
from pytorch_pretrained_bert.maxheap import heappush_max, nlargest


class TransformerXLBeamSearch:
    def __init__(self, model, bs, term_token, bw=None, max_len=50):
        self.model = model
        self.bs = bs
        self.bw = bw or bs
        self.max_len = max_len
        self.term_token = term_token

    def _topk(self, logprobs):
        b, v = logprobs.size()
        logprobs, indices2 = logprobs.topk(self.bw, dim=-1)  # [bs * bs]
        indices2 = indices2 + (torch.arange(b).to(logprobs.device) * v).long()[:, None]
        logprobs, indices = logprobs.view(-1).topk(min(self.bs, b * self.bw))  # [bs]
        indices = indices2.view(-1).index_select(0, indices)
        beam_indices = indices / v
        token_indices = indices % v
        return logprobs, beam_indices, token_indices

    def _logprobs(self, x, mask, next_token_restrictions):
        self.model.eval()
        with torch.no_grad():
            #bs = x.shape[0]
            #input_lens = mask.sum(1)
            logprobs, mems = self.model(x.view([-1, 1]).long())
            logprobs = logprobs.squeeze(1)
            # compute probability only on last token
            #output = output[torch.arange(bs).long(), input_lens - 1]
            #logits = self.model.lm_head(output)
            #logits = self._apply_token_mask(logits, next_token_restrictions, vocab_size=logits.size(-1))
            #logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            return logprobs, mems

    @staticmethod
    def _apply_token_mask(logits, next_token_restrictions, vocab_size):
        if all(a is None for a in next_token_restrictions):
            return logits

        token_mask = []
        for i in range(len(next_token_restrictions)):
            restrictions = next_token_restrictions[i]
            if restrictions is None:
                token_mask_i = torch.zeros((vocab_size,))
            else:
                token_mask_i = torch.full((vocab_size,), -np.inf)
                token_mask_i[restrictions] = 0.0
            token_mask.append(token_mask_i)

        return logits + torch.stack(token_mask, dim=0).to(logits.device)

    @staticmethod
    def _next_token_restrictions(restriction_ids, time_step, token_restrictions):
        token_masks = []
        for i in restriction_ids:
            token_restrictions_i = token_restrictions[i]
            if len(token_restrictions_i) <= time_step:
                token_masks.append(None)
                continue

            for t, token in enumerate(token_restrictions_i):
                if t == time_step:
                    token_masks.append(token)
                    break

        return token_masks

    def predict(self, x, mask, token_restrictions=None):

        #assert x.shape == mask.shape
        assert len(x.shape) == 1

        # 1. keep only what is needed.
        input_len = torch.sum(mask).item()
        x, mask = x.unsqueeze(0)[:, :input_len], mask.unsqueeze(0)[:, :input_len]

        # 2. Initialize token restrictions
        next_token_restrictions, restriction_indices = [None], torch.from_numpy(np.asarray([0])).to(x.device)
        if token_restrictions is not None:
            restriction_indices = torch.from_numpy(np.arange(len(token_restrictions))).to(x.device)
            next_token_restrictions = self._next_token_restrictions(restriction_ids=restriction_indices,
                                                                    time_step=0,
                                                                    token_restrictions=token_restrictions)

        predictions = []

        # 3. Replicate input to different token_restrictions
        x = x.expand([len(restriction_indices), x.shape[-1]])
        mask = mask.expand([len(restriction_indices), x.shape[-1]])

        def add_candidates(logprobs_i, x_i, mask_i, each_logprobs_i):
            """ Add finished candidates to max-heap """
            batch_size = len(logprobs_i)
            for j in range(batch_size):
                if logprobs_i[j] != -np.inf:
                    heappush_max(predictions, (np.exp(logprobs_i[j].item()),
                                               torch.masked_select(x_i[j], mask_i[j]).tolist()[input_len:],
                                               torch.masked_select(each_logprobs_i[j], mask_i[j]).tolist()[input_len:]))

        def apply_mask(boolean_mask, logprobs, x, mask, each_logprobs):
            logprobs_masked = logprobs[boolean_mask]
            x_masked = x[boolean_mask, :]
            mask_masked = mask[boolean_mask, :]
            each_logprobs_masked = each_logprobs[boolean_mask, :]
            return logprobs_masked, x_masked, mask_masked, each_logprobs_masked

        # 4. compute first beam-size token_indices
        logprobs, mems = self._logprobs(x, mask, next_token_restrictions)
        unfinished_logprobs, beam_indices, token_indices = self._topk(logprobs)
        restriction_indices = restriction_indices[beam_indices]
        term_mask = (unfinished_logprobs == -np.inf) | (token_indices == self.term_token).view(-1)

        # 5. initialise unfinished_{x|mask|each_logprobs}
        unfinished_x = x[beam_indices, :]
        unfinished_mask = mask[beam_indices, :]
        unfinished_each_logprobs = -np.inf * torch.ones_like(unfinished_x).type(unfinished_logprobs.type())

        # 6. append new indices and new mask
        unfinished_x = torch.cat([unfinished_x, token_indices.view([-1, 1])], dim=1)
        unfinished_mask = torch.cat([unfinished_mask, (1 - term_mask).view([-1, 1])], dim=1)
        unfinished_each_logprobs = torch.cat([unfinished_each_logprobs, unfinished_logprobs.view([-1, 1])], dim=1)

        # 7. split into finished and unfinished, add finished as candidates and update restriction_indices
        finished_logprobs, finished_x, finished_mask, finished_each_logprobs = apply_mask(term_mask,
                                                                                          unfinished_logprobs,
                                                                                          unfinished_x,
                                                                                          unfinished_mask,
                                                                                          unfinished_each_logprobs)

        unfinished_logprobs, unfinished_x, unfinished_mask, unfinished_each_logprobs = apply_mask(1 - term_mask,
                                                                                                  unfinished_logprobs,
                                                                                                  unfinished_x,
                                                                                                  unfinished_mask,
                                                                                                  unfinished_each_logprobs)

        restriction_indices = restriction_indices[1 - term_mask]

        add_candidates(finished_logprobs, finished_x, finished_mask, finished_each_logprobs)

        for i in range(self.max_len):
            # 8. compute token probs for unfinished_beam
            if token_restrictions is not None:
                next_token_restrictions = self._next_token_restrictions(restriction_ids=restriction_indices,
                                                                        time_step=i + 1,
                                                                        token_restrictions=token_restrictions)
            unfinished_logprobs_i = self._logprobs(unfinished_x, unfinished_mask, next_token_restrictions)

            # 9. add previous logprobs of unfinished_beams
            unfinished_logprobs_tmp = unfinished_logprobs_i + unfinished_logprobs.view([-1, 1])

            # 10. compute next_step indices
            unfinished_logprobs, beam_indices, token_indices = self._topk(unfinished_logprobs_tmp)
            restriction_indices = restriction_indices[beam_indices]
            term_mask = (unfinished_logprobs == -np.inf) | (token_indices == self.term_token).view(
                -1) if i < self.max_len - 1 else (token_indices >= 0).view(-1)

            # 11. re-align x and mask
            unfinished_x, unfinished_mask = unfinished_x[beam_indices, :], unfinished_mask[beam_indices, :]

            # 12. append new indices and update the mask
            unfinished_x = torch.cat([unfinished_x, token_indices.view([-1, 1])], dim=1)
            unfinished_mask = torch.cat([unfinished_mask, (1 - term_mask).view([-1, 1])], dim=1)
            unfinished_each_logprobs = torch.cat([unfinished_each_logprobs[beam_indices, :],
                                                  unfinished_logprobs_i[beam_indices, token_indices].view([-1, 1])],
                                                 dim=1)

            # 13. split into finished and unfinished beams and update restriction_indices
            finished_logprobs_i, finished_x_i, finished_mask_i, finished_each_logprobs_i = apply_mask(term_mask,
                                                                                                      unfinished_logprobs,
                                                                                                      unfinished_x,
                                                                                                      unfinished_mask,
                                                                                                      unfinished_each_logprobs)

            unfinished_logprobs, unfinished_x, unfinished_mask, unfinished_each_logprobs = apply_mask(1 - term_mask,
                                                                                                      unfinished_logprobs,
                                                                                                      unfinished_x,
                                                                                                      unfinished_mask,
                                                                                                      unfinished_each_logprobs)

            restriction_indices = restriction_indices[1 - term_mask]

            if term_mask.any():
                # 14. Add finished to predicted candidates
                add_candidates(finished_logprobs_i, finished_x_i, finished_mask_i, finished_each_logprobs_i)

            if term_mask.all():
                break

        return nlargest(self.bs, predictions)


if __name__ == '__main__':

    tokenizer = torch.load('./models/transfo_xl_vocab.bin')

    model_config = TransfoXLConfig(
        vocab_size_or_config_json_file=len(tokenizer),
        cutoffs=[20000, 40000, 200000],
        d_model=1024,
        d_embed=1024,
        n_head=16,
        d_head=64,
        d_inner=4096,
        div_val=4,
        pre_lnorm=False,
        n_layer=18,
        tgt_len=128,
        ext_len=0,
        mem_len=1600,
        clamp_len=1000,
        same_length=True,
        proj_share_all_but_first=True,
        attn_type=0,
        sample_softmax=-1,
        adaptive=True,
        tie_weight=True,
        dropout=0.2,
        dropatt=0.2,
        untie_r=True,
        init="normal",
        init_range=0.01,
        proj_init_std=0.01,
        init_std=0.02)

    model = TransfoXLLMHeadModel(config=model_config)
    model.load_pretrained_from_path("./models/transfo_xl_large.bin")
    model.eval()

    lm = TransformerXLBeamSearch(model=model, bs=5, term_token=0, bw=2, max_len=50)

    text_1 = "Who was Jim Henson ?"
    tokens_tensor_1 = np.asarray(tokenizer.transform(text_1), dtype=np.int32)
    mask = np.ones_like(tokens_tensor_1, dtype=np.uint8)

    x, mems = model(torch.LongTensor(tokens_tensor_1).view([-1, 1]))
    x, new_mems = model(torch.LongTensor(tokens_tensor_1).view([-1, 1]), mems=mems)

    lm.predict(x=torch.from_numpy(tokens_tensor_1), mask=torch.from_numpy(mask), token_restrictions=None)

