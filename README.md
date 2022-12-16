# FinABSA
[FinABSA](https://huggingface.co/amphora/FinABSA) is a T5-Large model trained for Aspect-Based Sentiment Analysis specifically for financial domains. The model was trained on the [SEntFiN 1.0](https://asistdl.onlinelibrary.wiley.com/doi/10.1002/asi.24634?af=R) and with several data augmentation tricks. Unlike traditional sentiment analysis models which predict a single sentiment label for each sentence, FinABSA has been trained to disambiguate sentences containing multiple aspects. By replacing the target aspect with a [TGT] token the model predicts the sentiment concentrating to the aspect.

## Comparision with FinBERT
In this sector, we compare FinABSA againset [FinBERT](https://github.com/ProsusAI/finBERT), the most widely used pretrained language model in the finance domain. As shown in the table below due to the structual limitations of the task itself FinBERT fails to correctly classify sentences with contradicting sentiments.

| SRC                                                | Model     | Prediction      |
| -------------------------------------------------- | --------- | --------------- |
| Tesla stocks dropped 42% while Samsung rallied.    | FinBERT   |  NEGATIVE       |
| [TGT] stocks dropped 42% while Samsung rallied.    | FinABSA   |  NEGATIVE       |
| Tesla stocks dropped 42% while [TGT] rallied.      | FinASBA   |  POSITIVE       |

## Supported Models
FinABSA is supported in two models, [T5-Large](https://huggingface.co/t5-large) and [DeBERTa-v3-Base](https://huggingface.co/microsoft/deberta-v3-base). T5 was trained using a prompt generation method similar to that of [Aspect Sentiment Quad Prediction as Paraphrase Generation](https://arxiv.org/abs/2110.00796). DeBERTa was trained using a conventional sequence classification objective. We observe that the DeBERTa model achieves a slightly higher accuracy compared to the T5 model. All models are available at [huggingface](https://huggingface.co/amphora).

## How To Use

### 1. Install Dependencies
```python
!git clone https://github.com/guijinSON/FinABSA.git
!pip install transformers
!pip install flair
```

### 2. Import & Run ABSA
```python
from FinABSA.FinABSA import ABSA
import pprint

absa = ABSA()
o = absa.run_absa(
                input_str = "Economic headwinds also add to uncertainties. Major companies, including Alphabet, Apple, Microsoft, and Meta, have indicated a \
                slowing pace of hiring for the rest of the year, suggesting mega-tech firms are bracing for a more uncertain economic outlook. \
                Tougher macroeconomic conditions are likely to lead to cuts in advertising budgets, while the squeeze on household disposable income through \
                inflation should weigh on discretionary consumer spending—both factors will hurt e-commerce and digital media companies. Meanwhile, reduced capital \
                expenditure and elevated inventories will likely weigh on semiconductor and hardware companies.."
                )
pprint.pprint(o)

# {'Alphabet': {'classification_output': 'NEGATIVE',
#              'logits': {'negative': 0.9970308542251587,
#                         'neutral': 0.0018199979094788432,
#                         'positive': 0.0011491376208141446}},
# 'Apple': {'classification_output': 'NEGATIVE',
#           'logits': {'negative': 0.9980792999267578,
#                      'neutral': 0.0013628738233819604,
#                     'positive': 0.000557745574042201}},
# 'Meta': {'classification_output': 'NEGATIVE',
#          'logits': {'negative': 0.9947644472122192,
#                     'neutral': 0.004664959851652384,
#                     'positive': 0.0005706017836928368}},
# 'Microsoft': {'classification_output': 'NEGATIVE',
#               'logits': {'negative': 0.9938719272613525,
#                          'neutral': 0.005691188853234053,
#                          'positive': 0.00043679968803189695}}}
```

#### OR

### 1. Directly import from HuggingFace
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("amphora/FinABSA")
tokenizer = AutoTokenizer.from_pretrained("amphora/FinABSA")
```

### 2. Generate your output
```python
input_str = "[TGT] stocks dropped 42% while Samsung rallied."
input = tokenizer(input_str, return_tensors='pt')
output = model.generate(**input, max_length=20)
print(tokenizer.decode(output[0]))

# The sentiment for [TGT] in the given sentence is NEGATIVE.
```

## Limitations
The model shows lower performance as the input sentence gets longer, mostly because the dataset used to train the model consists of source sentences with short sequences. To apply the model on longer sequences try [amphora/FinABSA-Longer](https://huggingface.co/amphora/FinABSA-Longer).

## Contact

Feel free to reach me at spthsrbwls123@yonsei.ac.kr
