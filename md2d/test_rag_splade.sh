from transformers import AutoConfig, AutoModel
from rag_splade import SpladeConfig, SpladeModel

AutoConfig.register("splade", SpladeConfig)
AutoModel.register(SpladeConfig, SpladeModel)

config = SpladeConfig()
model = SpladeModel(config)

model = SpladeModel.from_pretrained("/home/adityasv/multidoc2dial/splade/weights/dummy_to_convert_to_transformers")

import torch 
old = torch.load("/home/adityasv/multidoc2dial/splade/weights/distilsplade_max/pytorch_model.bin", map_location='cpu')
new = torch.load("/home/adityasv/multidoc2dial/splade/weights/dummy_to_convert_to_transformers/pytorch_model.bin", map_location='cpu')


# python /home/adityasv/multidoc2dial/md2d/rag_splade/convert_from_beir_to_transformers.py --checkpoint_dir ~/multidoc2dial/splade/weights/distilsplade_max --dump_path /home/adityasv/multidoc2dial/splade/weights/dummy_to_convert_to_transformers


import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from rag_splade import SpladeConfig, SpladeModel

class Splade(torch.nn.Module):
    def __init__(self, model_type_or_dir, agg="max"):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        assert agg in ("sum", "max")
        self.agg = agg
    
    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        if self.agg == "max":
            values, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
            return values
            # 0 masking also works with max because all activations are positive
        else:
            return torch.sum(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)


agg = "max"
model_type_or_dir = "/home/adityasv/multidoc2dial/splade/weights/distilsplade_max"
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)

model = Splade(model_type_or_dir, agg=agg)
model.eval()
tf_model = SpladeModel.from_pretrained("/home/adityasv/multidoc2dial/splade/weights/dummy_to_convert_to_transformers")
tf_model.eval()


doc = "Glass and Thermal Stress. Thermal Stress is created when one area of a glass pane gets hotter than an adjacent area. If the stress is too great then the glass will crack. The stress level at which the glass will break is governed by several factors."


a1 = model(**tokenizer(doc, return_tensors="pt")).squeeze() 
a2 = torch.nonzero(a1).squeeze()

print(a2, a1[a2])

b1 = tf_model(**tokenizer(doc, return_tensors="pt")).pooler_output.squeeze() 
b2 = torch.nonzero(b1).squeeze()
print(b2, b1[b2])

print(a1-b1, a2 - b2)