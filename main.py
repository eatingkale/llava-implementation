# imports
import torch
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from datasets import load_dataset
import os
import PIL 


# << dataset definition >>
class LLaVADataset(Dataset):
    def __init__(self, hf_dir="data/llava_instruct_150k.json", coco_dir=None): # TODO: replace None with a directory once downloaded images
        self.data = load_dataset("json", data_files=hf_dir)["train"]
        self.coco_img_dir = coco_dir
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx): # how we access an item of the dataset
        example = self.data[idx]
        # join local coco images with the example
        filename = example['image']
        if self.coco_img_dir is None:
            return example['conversations'] # TODO: fix this behavior -- is this the right behavior? what do we do when the image doesn't exist? 
                                            # when do we address this point? it certainly can't be at the time of accessing the data;
                                            # but as it stands now, all that's necessary to create a LLaVADataset is the json.
                                            # From brainstorming.ipynb, we saw that the actual number of images from coco that aligned
                                            # with the 'image' filename was less than 158k (approx. 100k). 
                                            # Where it's handled may not matter much, and the impact will just be that we leverage *less*
                                            # instruction-tuning data.
        img_path = os.path.join(self.coco_img_dir, filename)
        image = PIL.Image.open(img_path) if os.path.exists(img_path) else None # do we use None here or let Error get raised?
        example_conversation = example['conversations']
        return image, example_conversation


# << llava model definition >>
class LLaVAModel(nn.Module):
    def __init__(self, 
                 vision_checkpoint="google/siglip2-base-patch32-256", # patches of 32x32 and resize to at least 256 by 256 # transformers only take fixed size input
                 lm_checkpoint="LiquidAI/LFM2-350M"
                 ):
        # init vision enc
        self.vision_enc = AutoModel.from_pretrained(vision_checkpoint, device_map="auto").eval()
        self.vision_proc = AutoProcessor.from_pretrained(vision_checkpoint)

        # init LM
        self.lm = AutoModelForCausalLM.from_pretrained(
             lm_checkpoint,
             device_map="auto",
             torch_dtype="bfloat16",
             # attn_implementation="flash_attention_2" <- uncomment on compatible GPU
        ) # do we .eval() ?
        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_checkpoint)

        # dimensions
        self.v_dim = self.vision_enc.config.vision_config.hidden_size # not accounting for scenario where these attributes do not exist ..
        self.lm_dim = self.lm.config.hidden_size # assuming these attributes exist.
        
        # init projection
        # What initialization of weights? I know of xavier, he, and of course there is random? and zeros. the paper doesn't specify how they initialize; maybe let's look at their repo --> their repo 
        self.W = torch.zeros([self.v_dim, self.lm_dim], dtype=torch.float32)
        
    def forward(self, x, lang_embedding): 
        """ Forward pass
        Args:
            x : X_v in the paper
            lang_embedding : text instruction after tokenization and conversion to an embedding
        """
        # pseudocode ish
        # pass in attributes or just the model itself?
        # pass in the dimensionss (vision enc dim, language emb dim) as params or extract within the function?
        # pass in proj matrix or instantiate it here?

        z_v = self.vision_enc(x)
        h_v = self.W @ z_v
        all_tokens = torch.cat([h_v, lang_embedding], dim=1)
        llm_response = self.lm.get_response(all_tokens) # Is this the correct way to get response? lm.generate()?
        return llm_response # or maybe don't need response for this? or not sure

    # do we need to define train_step?
    # def train_step(W, other inputs):
    #   forward -> compute loss on ground truth -> backprop
    #   return new weights 

def main():
    print("Hello from llava-implementation!")

    dataset = LLaVADataset()
    print(dataset.data)
    print(dataset.coco_img_dir)
    example = dataset[0]
    print(example)
    

    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size = 8,
    #     num_workers = 1) # shape(batchsize, ... # todo: finish
    # for batch in dataloader:
    #     (batch_img, batch_text) = batch
    
    # training
    # "...fine-tune on the proposed LLaVA-Instruct-158K dataset for 3 epochs, with a learning rate of 2e-5 and a batch size of 32"


if __name__ == "__main__":
    main()


"""
===== Notes ====

====== image pipeline ====== 
(Image): .png/.jpg, local or remote 
    |
    | jpg -> Image: load image using PIL to get an Image
    | Image -> Tensor: do we use the Siglip Processor here? (autoprocessor??)
    |                   or something manual?
    V
(X_v)
    |
    V
(Vision Encoder): Siglip2 (modified CLIP) input needs to be a tensor? or image? shape?
    | 
    | inference
    V
(Visual Features)
    | 
    | project to LM space via W: W @ visual features
    | 
    V
(H_v): the LM-dim embedding.

====== Text ====== 
(instruct dataset): 'conversations' in data[0] Datasets
    | no change?
    V
(X_q): dict of the human-LM 'conversations' or instruction
    |
    | Tokenization?
    V
(Tokenized X_q)
    |
    | Going from tokenization to embedding =?
    | Embed_tokens = Embedding() in LFM2
    V
(H_q): language embedding of dim equal to LM input dim


====== Getting LM response ======
(H_v) (H_q)
    | Concatenate
    |
    V
(input to lm) = concat(h_v, h_q)
    |
    | pass through LM
    |
    V
(X_a: Language response)


====== evaluation or loss computation ======
(X_a)
| 
| comparison to ground truth
|
V


"""