"""
This is where the full implementation will be. 
"""

# imports
import transformers
from torch import DataLoader, Dataset
import torch


# dataset definition
class LLaVADataset(Dataset): #is this the correct superclass
    def __init__():
        temp = 1
        #init dataset params
        #more stuff
        # initialize self.lm, self.vision_enc, as LFM2 and siglip

# dataloader definition

# llava model definition
class LLaVAModel():
    def __init__():
        etc = etc # placeholder 
        # self.vision_enc = Siglip2 Model
        # self.lm = LFM2 model
        # v_dim = self.vision_enc.out_proj #or something like this 
        # lm_dim = self.lm.dimension #some attribute of the lm has the dimension
        # self.W = tensor with shape v_dim, lm_dim

    # inside this - forward?
    # loss
    # compute gradient
    # 
    def fwd(self, input, lang_tokens): 
        # pseudocode ish
        # pass in attributes or just the model itself?
        # pass in the dimensionss (vision enc dim, language emb dim) as params or extract within the function?
        # pass in proj matrix or instantiate it here?

        z_v = self.vision_enc(input)
        h_v = self.W @ z_v
        all_tokens = concatenate h_v + lang_tokens
        llm_response = self.lm.get_response(all_tokens)
        return llm_response # or maybe don't need response for this? or not sure




def train():
    placeholder = 1

def main():
    print("Hello from llava-implementation!")


if __name__ == "__main__":
    main()
