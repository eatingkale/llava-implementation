"""
This is where the full implementation will be.

See the end of the file for notes on the complete workflow:

"""

# imports
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch import DataLoader, Dataset
import torch.nn as nn
from datasets import load_dataset
import torch


# << dataset definition >>
class LLaVADataset(Dataset):
    def __init__():
        temp = 1
        #what normally goes in here, a transform? the function to find the image from 'image' ?
        # does tokenization go here????
    def __len__(self):
        temp = 1

    def __getitem__(self,idx):
        temp = 1


    # Example data point, corresponding to the way I loaded data in brainstorming.ipynb "load instruction tuning dataset"
    # data[0] =
    # {'id': '000000033471',
    #  'image': '000000033471.jpg',
    #  'conversations': [{'from': 'human',
    #    'value': '<image>\nWhat are the colors of the bus in the image?'},
    #   {'from': 'gpt', 'value': 'The bus in the image is white and red.'},
    #   {'from': 'human',
    #    'value': 'What feature can be seen on the back of the bus?'},
    #   {'from': 'gpt', 'value': 'The back of the bus features an advertisement.'},
    #   {'from': 'human',
    #    'value': 'Is the bus driving down the street or pulled off to the side?'},
    #   {'from': 'gpt',
    #    'value': 'The bus is driving down the street, which is crowded with people and other vehicles.'}]}


# << dataloader >>


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
        # What initialization of weights?
        self.W = torch.zeros([self.v_dim, self.lm_dim], dtype=torch.float32)
        
        # self.W = tensor with shape v_dim, lm_dim

    # inside this - forward?
    # loss / compute gradient?
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
        llm_response = self.lm.get_response(all_tokens)
        return llm_response # or maybe don't need response for this? or not sure

# << def trainer/training arguments >>
    # Do we do this in main()?
    # Notes from HF:
        # Trainer is a complete training and evaluation loop for Transformers’ PyTorch models. 
        # Pass your model, dataset, preprocessor, and TrainingArguments to Trainer, and call train() to start training.
    # --> Probably don't need to make a custom class
    # do we need to define train_step?
    # def train_step(W, other inputs):
    #   forward -> compute loss on ground truth -> backprop
    #   return new weights 

def main():
    print("Hello from llava-implementation!")

    dataloader = DataLoader() # shape(batchsize, ... # todo: finish
    for batch in dataloader:
        (batch_img, batch_text) = batch
    
    # training


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