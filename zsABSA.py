import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification

class ZS_ABSA():
    def __init__(self,
                 MaskedLM_ckpt='roberta-large',
                 ClassifierLM_ckpt="ProsusAI/finbert",
                 Classifier_classification = {0:"POSITIVE", 1:"NEGATIVE", 2:"NEUTRAL"},
                 prompt="Given this [TGT] shares are likely to [MSK].",
                 device='cpu'
                 ):
        
        self.MaskedLM = AutoModelForMaskedLM.from_pretrained(MaskedLM_ckpt)
        _ = self.MaskedLM.eval()
        self.MaskedLM_tok = AutoTokenizer.from_pretrained(MaskedLM_ckpt)

        self.ClassifierLM = AutoModelForSequenceClassification.from_pretrained(ClassifierLM_ckpt)
        _ = self.ClassifierLM.eval()
        self.ClassifierLM_tok = AutoTokenizer.from_pretrained(ClassifierLM_ckpt)
        self.Classifier_classification = Classifier_classification

        self.prompt = prompt
        self.device = device
    
    def run_ABSA(self,input_str, tgt):
        prompt = self.prompt.replace('[TGT]',tgt).replace('[MSK]',self.MaskedLM_tok.mask_token)
        input_str = input_str + self.MaskedLM_tok.sep_token + prompt
        MaskedLMOutput = self.prompt_mask_filling(input_str)
        ClassifierLMOutput = self.prompt_sentiment_analysis(MaskedLMOutput['output_sent'])

        return {
            'src':input_str,
            'target':tgt,
            'MaskedLMOutput': MaskedLMOutput,
            'ClassifierLMOutput':ClassifierLMOutput
        }

    def prompt_mask_filling(self, input_str):
        input = self.MaskedLM_tok(input_str,return_tensors='pt')
        input_ids = input['input_ids']
        mask_token_idx = torch.where(input_ids==self.MaskedLM_tok.mask_token_id,True,False).squeeze().nonzero().item()

        output_logits = self.MaskedLM(
                                input_ids = input_ids.to(self.device),
                                attention_mask = input['attention_mask'].to(self.device)
                                ).logits[0]
        
        input_ids = input_ids.squeeze()
        output = torch.argmax(output_logits,dim=-1).squeeze()
        input_ids[mask_token_idx] = output[mask_token_idx]
        output_sent = self.MaskedLM_tok.decode(input_ids)

        output_logits = torch.topk(F.softmax(output_logits[-3],dim=0),3,dim=0)
        return {
            "output_sent":output_sent,
            "output_logit_top3": list(output_logits.values.detach().numpy()),
            "output_indices_top3":list(output_logits.indices.detach().numpy())
        }
    
    def prompt_sentiment_analysis(self,input_str):
        input = self.ClassifierLM_tok(input_str,return_tensors='pt')
        output = self.ClassifierLM(
                                    input_ids = input['input_ids'].to(self.device),
                                    attention_mask = input['attention_mask'].to(self.device)
                                    ).logits
        output = torch.argmax(output,dim=-1)
        output_class = self.Classifier_classification[int(output)]
        return output_class
