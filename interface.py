import torch
import os
import sys
from trainer import ContinuousTrainer
from response_generator import ResponseGenerator

class LLMInterface:
    def __init__(self, model_dir="custom_llm", tokenizer_type="bpe"):
        self.model_dir = model_dir
        self.tokenizer_type = tokenizer_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainer = None
        self.responder = None
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024
            print(f"GPU Memory: {memory_gb}GB")
        print(f"Tokenizer type: {tokenizer_type}")
    
    def init_trainer(self):
        if self.trainer is None:
            self.trainer = ContinuousTrainer(self.model_dir, self.device, self.tokenizer_type)
        return self.trainer
    
    def init_responder(self):
        if self.responder is None:
            if os.path.exists(os.path.join(self.model_dir, 'config.json')):
                self.responder = ResponseGenerator(self.model_dir, self.device)
            else:
                print("No trained model found. Train a model first.")
                return None
        return self.responder
    
    def training_menu(self):
        trainer = self.init_trainer()
        
        while True:
            print("\n" + "="*60)
            print("TRAINING MODE")
            print("="*60)
            print("1. Train from text file")
            print("2. Train from direct text input")
            print("3. Continue training with new data")
            print("4. Check training status")
            print("5. Back to main menu")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                file_path = input("Enter file path: ").strip()
                if os.path.exists(file_path):
                    texts = trainer.load_data_from_file(file_path)
                    if texts:
                        epochs = int(input("Enter number of epochs (default 10): ") or 10)
                        print(f"\nStarting training on {len(texts)} text chunks...")
                        trainer.train_continuous(texts, epochs=epochs)
                        print("Training completed!")
                else:
                    print("File not found!")
            
            elif choice == '2':
                print("Enter your text (type 'END' on a new line to finish):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    lines.append(line)
                
                if lines:
                    text = "\n".join(lines)
                    chunks = [text[i:i + 512] for i in range(0, len(text), 256)]
                    chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 10]
                    epochs = int(input("Enter number of epochs (default 5): ") or 5)
                    print(f"\nStarting training on {len(chunks)} text chunks...")
                    trainer.train_continuous(chunks, epochs=epochs)
                    print("Training completed!")
                else:
                    print("No text entered!")
            
            elif choice == '3':
                file_path = input("Enter new data file path: ").strip()
                if os.path.exists(file_path):
                    new_texts = trainer.load_data_from_file(file_path)
                    if new_texts:
                        epochs = int(input("Enter number of epochs (default 5): ") or 5)
                        print(f"\nContinuing training with {len(new_texts)} new text chunks...")
                        trainer.train_on_new_data(new_texts, epochs=epochs)
                        print("Training completed!")
                else:
                    print("File not found!")
            
            elif choice == '4':
                if trainer.model:
                    vocab_size = getattr(trainer.tokenizer, 'vocab_size_actual', len(getattr(trainer.tokenizer, 'vocab', {})))
                    print(f"\nModel Status:")
                    print(f"- Model exists: Yes")
                    print(f"- Current epoch: {trainer.current_epoch}")
                    print(f"- Vocabulary size: {vocab_size}")
                    print(f"- Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
                    print(f"- Model directory: {trainer.model_dir}")
                    print(f"- Tokenizer type: {trainer.tokenizer_type}")
                else:
                    print("No model loaded")
            
            elif choice == '5':
                break
            
            else:
                print("Invalid choice!")
    
    def response_menu(self):
        responder = self.init_responder()
        if responder is None:
            return
        
        while True:
            print("\n" + "="*60)
            print("RESPONSE MODE - Q&A")
            print("="*60)
            print("1. Ask a question")
            print("2. Chat mode (continuous conversation)")
            print("3. Generate text from prompt")
            print("4. Model information")
            print("5. Back to main menu")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                question = input("\nEnter your question: ").strip()
                if question:
                    print("\nGenerating response...")
                    response = responder.generate_response(question)
                    print(f"\nQ: {question}")
                    print(f"A: {response}")
            
            elif choice == '2':
                print("\nChat Mode - Type 'exit' to return to menu")
                print("-" * 50)
                while True:
                    user_input = input("\nYou: ").strip()
                    if user_input.lower() == 'exit':
                        break
                    if user_input:
                        response = responder.generate_response(user_input)
                        print(f"AI: {response}")
            
            elif choice == '3':
                prompt = input("\nEnter prompt for text generation: ").strip()
                if prompt:
                    max_length = int(input("Enter max response length (default 100): ") or 100)
                    print("\nGenerating text...")
                    response = responder.generate_text(prompt, max_length=max_length)
                    print(f"\nPrompt: {prompt}")
                    print(f"Generated: {response}")
            
            elif choice == '4':
                info = responder.get_model_info()
                print(f"\nModel Information:")
                for key, value in info.items():
                    print(f"- {key}: {value}")
            
            elif choice == '5':
                break
            
            else:
                print("Invalid choice!")
    
    def main_menu(self):
        while True:
            print("\n" + "="*60)
            print("CUSTOM LLM SYSTEM")
            print("="*60)
            print("1. Training Mode")
            print("2. Response Mode (Q&A)")
            print("3. System Information")
            print("4. Exit")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                self.training_menu()
            
            elif choice == '2':
                self.response_menu()
            
            elif choice == '3':
                self.show_system_info()
            
            elif choice == '4':
                print("Goodbye!")
                sys.exit(0)
            
            else:
                print("Invalid choice!")
    
    def show_system_info(self):
        print(f"\nSystem Information:")
        print(f"- Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"- GPU: {torch.cuda.get_device_name()}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024 // 1024
            print(f"- GPU Memory: {memory_gb}GB")
        print(f"- Model Directory: {self.model_dir}")
        print(f"- Model Exists: {os.path.exists(os.path.join(self.model_dir, 'config.json'))}")
        print(f"- Tokenizer Type: {self.tokenizer_type}")

def main():
    print("Initializing Custom LLM System...")
    tokenizer_type = input("Choose tokenizer (bpe/sentencepiece) [default: bpe]: ").strip().lower() or "bpe"
    interface = LLMInterface(tokenizer_type=tokenizer_type)
    interface.main_menu()

if __name__ == "__main__":
    main()