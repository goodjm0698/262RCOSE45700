"""BERT model/input adapter (role of /root/ViT/vit_pilot/vit.py).

Mask handling follows BERT/step4_compare.py: create_bidirectional_mask is patched to a
passthrough and a 4D additive BF16 mask [B,1,L,L] is built on CPU and sent to the NPU.
"""
import hashlib
import json

MODEL = 'textattack/bert-base-uncased-SST-2'
SEQ_LEN = 128   # 48/64 fail to compile on NPU; never use them.
LABELS = ['negative', 'positive']

# Same 16 sentences as BERT/step4_compare.py (order preserved).
SENTENCES = [
    "a stirring , funny and finally transporting re-imagining of a classic story",
    "the film is strictly routine and never surprises",
    "one of the best performances of the year , hands down",
    "it 's so laddish and juvenile , only teenage boys could possibly find it funny",
    "a masterpiece of quiet , devastating storytelling",
    "the plot is paper thin and the dialogue is worse",
    "an absolute delight from start to finish",
    "painfully slow and utterly forgettable",
    "the acting elevates an otherwise ordinary script",
    "i wanted to walk out after twenty minutes",
    "visually stunning with a soundtrack to match",
    "a bland , lifeless retread of better films",
    "heartfelt , sincere , and genuinely moving",
    "the jokes land with a thud every single time",
    "an ambitious film that mostly succeeds",
    "a complete waste of a talented cast",
]


def patch_mask():
    import transformers.models.bert.modeling_bert as bert_mod

    def passthrough_mask(*args, **kwargs):
        return kwargs['attention_mask'] if 'attention_mask' in kwargs else args[2]
    bert_mod.create_bidirectional_mask = passthrough_mask


def additive_mask(mask2d, dtype):
    import torch
    neg = torch.finfo(dtype).min
    m = (1.0 - mask2d[:, None, None, :].to(torch.float32)) * neg       # [B,1,1,L]
    return m.expand(-1, 1, SEQ_LEN, -1).contiguous().to(dtype)         # [B,1,L,L]


def encode(tok, sentences):
    """CPU tensors: real sentences padded to 128, BF16 additive 4D mask."""
    import torch
    x = tok(sentences, padding='max_length', max_length=SEQ_LEN, truncation=True, return_tensors='pt')
    return dict(input_ids=x['input_ids'], token_type_ids=x['token_type_ids'],
                attention_mask=additive_mask(x['attention_mask'], torch.bfloat16),
                valid_tokens=[int(v) for v in x['attention_mask'].sum(1)])


def inputs_hash(inputs):
    h = hashlib.sha256()
    for k in ['input_ids', 'token_type_ids', 'attention_mask']:
        t = inputs[k].contiguous()
        h.update(json.dumps([k, str(t.dtype), list(t.shape)]).encode())
        h.update(t.view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def load_model():
    """BF16 eval model on CPU; caller moves it to the device."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    patch_mask()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(torch.bfloat16).eval()
    return tok, model
