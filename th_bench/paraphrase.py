import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
import tqdm
import json
from nltk.tokenize import sent_tokenize
from six.moves import cPickle as pkl
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained('google/t5-v1_1-xxl')
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        torch.cuda.empty_cache()
#        print(torch.cuda.memory_summary())
        self.model.cuda()
        self.model.eval()

    def paraphrase(self, input_text, lex_diversity, order_diversity, prefix="", sent_interval=1, **kwargs):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [0, 20, 40, 60, 80, 100], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [0, 20, 40, 60, 80, 100], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.replace("\n", " ").split())
        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        prefix = " ".join(prefix.split())
        prefix_sentences = sent_tokenize(prefix)
        output_text = ""
        prefix_output = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            import time
            time.sleep(1)
            torch.cuda.empty_cache()
            time.sleep(1)
            torch.cuda.empty_cache()
#            print(torch.cuda.memory_summary())
            print ("sent_idx", sent_idx, "sent_inter + sent_idx", sent_interval + sent_idx )
            curr_sent_window = " ".join(sentences[sent_idx:sent_idx + sent_interval])
            prefix_window = " ".join(sentences[sent_idx - (4 if sent_idx > 4 else sent_idx):sent_idx + sent_interval + (4 if (sent_idx + sent_interval + 4) < len(sentences) else (len(sentences) - sent_idx))])
            n = 0
            while len(prefix_window.split()) > 256:
                n += 1
                prefix_window = " ".join(prefix_sentences[sent_idx - ((4 - n) if sent_idx > 1 else sent_idx): sent_idx])
            prefix_output = " ".join(prefix_output.replace("\n", " ").split())
            prefix_output = " ".join(prefix_output.split())
            prefix_output_sentences = sent_tokenize(prefix_output)
            prefix_output_window = " ".join(prefix_output_sentences[sent_idx - (4 if sent_idx > 4 else sent_idx):sent_idx])
            n = 0
            while len(prefix_output_window.split()) > 256:
                n +=1
                prefix_output_window = " ".join(prefix_output_sentences[sent_idx - ((4 - n) if sent_idx > 1 else sent_idx):sent_idx])
            if len(curr_sent_window.split()) > 256:
                curr_sent_window = " ".join(curr_sent_window.split()[:256]) + "."
            print ("length", len(curr_sent_window.split()), len(prefix_window.split()), len(prefix_output_window.split()))
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix_window}"
            final_input_text += f" {prefix_output_window}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix_output += " " + outputs[0]
            output_text += " " + outputs[0]
            torch.cuda.empty_cache()

        return output_text

def run_attack_dipper(data, model="kalpeshk2011/dipper-paraphraser-xxl", P=1):
    dp = DipperParaphraser(model=model)
    out = [[] for j in range(P)]

    for idx, dd in tqdm.tqdm(enumerate(data['text']), total=len(data['text'])):
        torch.cuda.empty_cache()
 #       print(torch.cuda.memory_summary())
        if data['label'][idx] == 0:
            continue
        input_text = data['text'][idx]
        prompt = "" # this goes as context/prefix to paraphrase in 1st round

        for j in range(P):
            if j == P - 1:
                order_diversity = 60
            else:
                order_diversity = 0

            output = dp.paraphrase(input_text, lex_diversity=60, order_diversity=order_diversity, prefix=prompt, do_sample=True, top_p=0.75, top_k=None, max_length=750)
            out[j].append(output)

            if idx == 0:
                print("\n\nInput:", input_text)
                print("\n\nOutput:", j, output)

            prompt = input_text  # use current input as prefix for next round
            input_text = output  # recursive paraphrase. current output goes as input in next round.
        data['text'][idx] = output
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    if not os.path.exists("results"):
        os.mkdir("results")
    with open(f'results/pp_{P}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return data
