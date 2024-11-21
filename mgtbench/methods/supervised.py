import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
from ..auto import BaseDetector
from ..loading import load_pretrained_supervise
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import Trainer, TrainingArguments, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class SupervisedDetector(BaseDetector):
    def __init__(self, name, **kargs) -> None:
        super().__init__(name)
        self.model = kargs.get('model', None)
        self.tokenizer = kargs.get('tokenizer', None)
        if not self.model or not  self.tokenizer:
            model_name_or_path = kargs.get('model_name_or_path', None)
            if not model_name_or_path :
                raise ValueError('You should pass the model_name_or_path or a model instance, but none is given')
            quantitize_bit = kargs.get('load_in_k_bit', None)
            self.model, self.tokenizer = load_pretrained_supervise(model_name_or_path, kargs,quantitize_bit)
        if not isinstance(self.model, PreTrainedModel) or not isinstance(self.tokenizer, PreTrainedTokenizerBase):
            raise ValueError('Expect PreTrainedModel, PreTrainedTokenizer, got', type(self.model), type(self.tokenizer))
        
    def detect(self, text, **kargs):
        disable_tqdm = kargs.get('disable_tqdm', False)
        result = []
        if not isinstance(text, list):
            text = [text]
        n_positions = self.model.config.max_position_embeddings
        if n_positions<1000:
            n_positions=512
        else:
            n_positions=4096
            
        num_labels = self.model.config.num_labels
        # TODO: combine the two cases use inner loop if
        if num_labels == 2:
            pos_bit=1
            for batch in tqdm(DataLoader(text), disable=disable_tqdm):
                with torch.no_grad():
                    tokenized = self.tokenizer(
                        batch,
                        max_length=n_positions,
                        return_tensors="pt",
                        truncation = True
                    ).to(self.model.device)
                    result.append(self.model(**tokenized).logits.softmax(-1)[:, pos_bit].item())
        else:
            for batch in tqdm(DataLoader(text), disable=disable_tqdm):
                with torch.no_grad():
                    tokenized = self.tokenizer(
                        batch,
                        max_length=n_positions,
                        return_tensors="pt",
                        truncation = True
                    ).to(self.model.device)
                    result.append(torch.argmax(self.model(**tokenized).logits, dim=-1).item())

        return result if isinstance(text, list) else result[0]
    
    def finetune(self, data, config):
        if config.pos_bit == 0:
            data['label'] = [1 if label == 0 else 0 for label in data['label']]

        # Tokenize the data
        train_encodings = self.tokenizer(data['text'], truncation=True, padding=True)
        train_dataset = CustomDataset(train_encodings, data['label'])
        print(config.need_save)
        # Define the training arguments
        training_args = TrainingArguments(
            output_dir=config.save_path,              # Output directory
            num_train_epochs=config.epochs,           # Number of epochs
            per_device_train_batch_size=config.batch_size,  # Batch size
            save_strategy="epoch" if config.need_save else 'no', # Save after each epoch
            logging_dir='./logs',                    # Directory for logs
            logging_steps=50,                       # Log every 100 steps
            weight_decay=0.01,                       # Weight decay
            learning_rate=config.lr,                      # Learning rate
            save_total_limit=2,                      # Limit to save only the best checkpoints
            gradient_accumulation_steps=config.gradient_accumulation_steps
            # load_best_model_at_end=True if config.need_save else False  # Save best model
        )

        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
            optimizers=(AdamW(self.model.parameters(), lr=config.lr), None)  # Optimizer, lr_scheduler
            # do not use torch.optim.AdamW, will cause nan
        )

        # Train the model
        trainer.train()

        # Save the model if needed
        # if config.need_save:
        #     self.model.save_pretrained(f'{config.save_path}/{config.name}')