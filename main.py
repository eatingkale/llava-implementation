"""
This is where the full implementation will be.

See the end of the file for notes on the complete workflow:

"""

# imports
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch import DataLoader, Dataset
from datasets import load_dataset
import torch


# << dataset definition >>
    # class LLaVADataset(Dataset): #is this the correct superclass
    #     def __init__():
    #         temp = 1
    #         #what normally goes in here, a transform? the function to find the image from 'image' ?
    #         # does tokenization go here????

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
class LLaVAModel(): # what do we inherit?
    def __init__(self, 
                 vision_checkpoint="google/siglip2-base-patch32-256", 
                 lm_checkpoint="LiquidAI/LFM2-350M"
                 ): #do we add a Config or other Arguments/args into this initialization?
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
        # init projection
        self.W = torch.tensor()
        # do we need to get dimensions?
        # v_dim = self.vision_enc.out_proj #or something like this 
        # lm_dim = self.lm.dimension #some attribute of the lm has the dimension
        # self.W = tensor with shape v_dim, lm_dim

    # inside this - forward?
    # loss / compute gradient?
    # Does transformers abstract any of this away. with a Training class?
    def fwd(self, input, lang_embedding): 
        # pseudocode ish
        # pass in attributes or just the model itself?
        # pass in the dimensionss (vision enc dim, language emb dim) as params or extract within the function?
        # pass in proj matrix or instantiate it here?

        z_v = self.vision_enc(input)
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

def main():
    print("Hello from llava-implementation!")
    
    #Training
    train_args = TrainingArguments(
        ...
    )

    trainer = Trainer(
        ...,
        args = train_args,
    )
    trainer.train()


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
(instruct dataset): 'conversations' 

(X_q)

(H_q)


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
| 
|
V
to be continued ..

"""