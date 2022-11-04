from flair.data import Sentence
from flair.models import SequenceTagger

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn.functional as F

class ABSA():
    def __init__(self, 
                 ckpt_path="amphora/FinABSA",
                 NER_tag_list = ['ORG']
                 ):
        
        self.ABSA = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.tagger = SequenceTagger.load('ner')

        self.NER_tag_list = NER_tag_list

    def run_absa(self,input_str):
        tgt_entities = self.retrieve_target(input_str)
        output = {e : self.run_single_absa(input_str,e) for e in tgt_entities}
        return output

    def run_single_absa(self,input_str,tgt):
        input_str = input_str.replace(tgt, '[TGT]')
        input = self.tokenizer(input_str,return_tensors='pt')

        output = self.ABSA.generate(
                                    **input,
                                    max_length=20,
                                    output_scores=True,
                                    return_dict_in_generate=True
                                    )
        
        classification_output = self.tokenizer.convert_ids_to_tokens(
                                                    int(output['sequences'][0][-4])
                                                    )
        logits = F.softmax(output['scores'][-4][:,-3:],dim=1)[0]
        
        return {
                "classification_output": classification_output,
                "logits": 
                {
                    'positive': float(logits[0]),
                    'negative': float(logits[1]),
                    'neutral':  float(logits[2])
                }
        }

    def retrieve_target(self,input_str):
        sentence = Sentence(input_str)
        self.tagger.predict(sentence)
        entities = [entity.text for entity in sentence.get_spans('ner') if entity.tag in self.NER_tag_list]
        return entities
