import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import gc
from tqdm import tqdm
from model import CustomLLM, create_tokenizer, save_model, load_model

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            tokens = tokenizer.encode(text, max_length)
            if len(tokens) > 1:
                if len(tokens) < max_length:
                    pad_token = getattr(tokenizer, 'vocab', {}).get('<PAD>', 0)
                    tokens.extend([pad_token] * (max_length - len(tokens)))
                self.examples.append(tokens)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        tokens = self.examples[idx]
        return torch.tensor(tokens, dtype=torch.long)

class ContinuousTrainer:
    def __init__(self, model_dir="custom_llm", device=None, tokenizer_type="bpe"):
        self.model_dir = model_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer_type = tokenizer_type
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.current_epoch = 0
        self.total_loss = 0.0
        self.step_count = 0
        
        self.setup_gpu()
        self.load_or_create_model()
        
    def setup_gpu(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024
            if gpu_memory_gb >= 6:
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.scaler = None
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {gpu_memory_gb}GB")
            print(f"Mixed Precision: {'Enabled' if self.scaler else 'Disabled'}")
        else:
            self.scaler = None
            print("Using CPU")
    
    def load_or_create_model(self):
        if os.path.exists(os.path.join(self.model_dir, 'config.json')):
            print("Loading existing model...")
            self.model, self.tokenizer = load_model(self.model_dir, self.device)
            
            checkpoint_path = os.path.join(self.model_dir, 'checkpoint.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.current_epoch = checkpoint.get('epoch', 0)
                print(f"Resuming from epoch {self.current_epoch}")
        else:
            print("No existing model found. Will create new model when training starts.")
    
    def create_new_model(self, texts, vocab_size=10000):
        print("Creating new model...")
        self.tokenizer = create_tokenizer(self.tokenizer_type, vocab_size)
        
        if hasattr(self.tokenizer, 'train'):
            if self.tokenizer_type == "sentencepiece":
                model_path = os.path.join(self.model_dir, "tokenizer_temp")
                os.makedirs(self.model_dir, exist_ok=True)
                self.tokenizer.train(texts, model_path)
            else:
                self.tokenizer.train(texts)
        
        actual_vocab_size = getattr(self.tokenizer, 'vocab_size_actual', len(getattr(self.tokenizer, 'vocab', {})))
        
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024 if self.device.type == 'cuda' else 4
        
        if gpu_memory_gb >= 24:
            config = {"d_model": 1024, "n_layers": 12, "n_heads": 16, "d_ff": 4096}
        elif gpu_memory_gb >= 12:
            config = {"d_model": 768, "n_layers": 8, "n_heads": 12, "d_ff": 3072}
        elif gpu_memory_gb >= 8:
            config = {"d_model": 512, "n_layers": 6, "n_heads": 8, "d_ff": 2048}
        else:
            config = {"d_model": 256, "n_layers": 4, "n_heads": 8, "d_ff": 1024}
        
        self.model = CustomLLM(
            vocab_size=actual_vocab_size,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=config["d_ff"],
            max_seq_len=256,
            dropout=0.1
        )
        
        self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Vocabulary size: {actual_vocab_size}")
    
    def setup_training(self, learning_rate=3e-4):
        if self.model is None:
            raise ValueError("Model not initialized. Call create_new_model() first.")
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        checkpoint_path = os.path.join(self.model_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded")
    
    def prepare_data(self, texts, batch_size=None):
        if batch_size is None:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024 if self.device.type == 'cuda' else 4
            if gpu_memory_gb <= 4:
                batch_size = 4
            else:
                batch_size = min(64, max(4, gpu_memory_gb * 4))
        
        dataset = TextDataset(texts, self.tokenizer, max_length=256)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=self.device.type == 'cuda',
            num_workers=0,
            persistent_workers=False
        )
        
        return dataloader
    
    def train_step(self, batch):
        batch = batch.to(self.device, non_blocking=True)
        
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                logits, loss = self.model(batch, targets=batch)
        else:
            logits, loss = self.model(batch, targets=batch)
        
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.scheduler.step()
        
        return loss.item()
    
    def train_continuous(self, texts, epochs=10, save_every=100):
        if self.model is None:
            self.create_new_model(texts)
        
        self.setup_training()
        
        dataloader = self.prepare_data(texts)
        
        print(f"Training on {len(dataloader.dataset)} examples")
        print(f"Batch size: {dataloader.batch_size}")
        print(f"Steps per epoch: {len(dataloader)}")
        
        self.model.train()
        
        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                progress_bar = tqdm(dataloader, desc=f'Epoch {self.current_epoch + epoch + 1}')
                
                for step, batch in enumerate(progress_bar):
                    loss = self.train_step(batch)
                    epoch_loss += loss
                    self.step_count += 1
                    
                    avg_loss = epoch_loss / (step + 1)
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                        'step': self.step_count
                    })
                    
                    if self.step_count % save_every == 0:
                        self.save_checkpoint(avg_loss)
                    
                    if self.device.type == 'cuda' and step % 50 == 0:
                        torch.cuda.empty_cache()
                
                self.current_epoch += 1
                avg_epoch_loss = epoch_loss / len(dataloader)
                print(f'Epoch {self.current_epoch} completed. Loss: {avg_epoch_loss:.4f}')
                
                self.save_checkpoint(avg_epoch_loss)
                
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current state...")
            self.save_checkpoint(avg_loss if 'avg_loss' in locals() else 0.0)
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    def train_on_new_data(self, new_texts, epochs=5):
        print(f"Training on {len(new_texts)} new text samples...")
        
        if self.model is None:
            self.create_new_model(new_texts)
        else:
            print("Expanding vocabulary with new data...")
            old_vocab_size = getattr(self.tokenizer, 'vocab_size_actual', len(getattr(self.tokenizer, 'vocab', {})))
            
            if hasattr(self.tokenizer, 'train'):
                if self.tokenizer_type == "sentencepiece":
                    model_path = os.path.join(self.model_dir, "tokenizer_temp_new")
                    self.tokenizer.train(new_texts, model_path)
                else:
                    self.tokenizer.train(new_texts)
            
            new_vocab_size = getattr(self.tokenizer, 'vocab_size_actual', len(getattr(self.tokenizer, 'vocab', {})))
            
            if new_vocab_size > old_vocab_size:
                print(f"Vocabulary expanded from {old_vocab_size} to {new_vocab_size}")
                self.expand_model_vocab()
        
        self.train_continuous(new_texts, epochs=epochs)
    
    def expand_model_vocab(self):
        old_embedding = self.model.token_embedding.weight.data
        old_lm_head = self.model.lm_head.weight.data
        
        new_vocab_size = getattr(self.tokenizer, 'vocab_size_actual', len(getattr(self.tokenizer, 'vocab', {})))
        
        new_embedding = nn.Embedding(new_vocab_size, self.model.d_model)
        new_lm_head = nn.Linear(self.model.d_model, new_vocab_size, bias=False)
        
        new_embedding.weight.data[:old_embedding.size(0)] = old_embedding
        new_lm_head.weight.data[:old_lm_head.size(0)] = old_lm_head
        
        self.model.token_embedding = new_embedding
        self.model.lm_head = new_lm_head
        self.model.lm_head.weight = self.model.token_embedding.weight
        self.model.vocab_size = new_vocab_size
        
        self.model.to(self.device)
    
    def save_checkpoint(self, loss):
        save_model(self.model, self.tokenizer, self.model_dir, 
                  self.optimizer, self.current_epoch, loss)
        print(f"Checkpoint saved at epoch {self.current_epoch}")
    
    def load_data_from_file(self, file_path, chunk_size=512):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if '.' in text:
                sentences = text.split('.')
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk) + len(sentence) + 1 > chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + "."
                    else:
                        current_chunk += " " + sentence + "." if current_chunk else sentence + "."
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
                chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 10]
            
            print(f"Loaded {len(chunks)} chunks from {file_path}")
            return chunks
            
        except FileNotFoundError:
            print(f"File {file_path} not found")
            return []
        except Exception as e:
            print(f"Error reading file: {e}")
            return []

def main():
    tokenizer_type = input("Choose tokenizer (bpe/sentencepiece) [default: bpe]: ").strip().lower() or "bpe"
    trainer = ContinuousTrainer(tokenizer_type=tokenizer_type)
    
    while True:
        print("\n" + "="*50)
        print("CUSTOM LLM CONTINUOUS TRAINER")
        print("="*50)
        print("1. Train from file")
        print("2. Train from text input")
        print("3. Continue training with new data")
        print("4. Check model status")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            file_path = input("Enter file path: ").strip()
            if os.path.exists(file_path):
                texts = trainer.load_data_from_file(file_path)
                if texts:
                    epochs = int(input("Enter number of epochs (default 10): ") or 10)
                    trainer.train_continuous(texts, epochs=epochs)
            else:
                print("File not found!")
        
        elif choice == '2':
            print("Enter your text (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            
            if lines:
                text = "\n".join(lines)
                chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
                epochs = int(input("Enter number of epochs (default 5): ") or 5)
                trainer.train_continuous(chunks, epochs=epochs)
        
        elif choice == '3':
            file_path = input("Enter new data file path: ").strip()
            if os.path.exists(file_path):
                new_texts = trainer.load_data_from_file(file_path)
                if new_texts:
                    epochs = int(input("Enter number of epochs (default 5): ") or 5)
                    trainer.train_on_new_data(new_texts, epochs=epochs)
            else:
                print("File not found!")
        
        elif choice == '4':
            if trainer.model:
                vocab_size = getattr(trainer.tokenizer, 'vocab_size_actual', len(getattr(trainer.tokenizer, 'vocab', {})))
                print(f"Model exists: Yes")
                print(f"Current epoch: {trainer.current_epoch}")
                print(f"Vocabulary size: {vocab_size}")
                print(f"Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
                print(f"Tokenizer type: {trainer.tokenizer_type}")
            else:
                print("No model loaded")
        
        elif choice == '5':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()