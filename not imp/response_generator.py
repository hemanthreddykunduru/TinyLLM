import torch
import torch.nn.functional as F
import re
import numpy as np
from model import load_model

class ResponseGenerator:
    def __init__(self, model_dir, device=None):
        self.model_dir = model_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.tokenizer_type = None
        
        self.load_model()
        
    def load_model(self):
        try:
            self.model, self.tokenizer = load_model(self.model_dir, self.device)
            
            # Determine tokenizer type
            if hasattr(self.tokenizer, 'sp') and self.tokenizer.sp is not None:
                self.tokenizer_type = 'sentencepiece'
            else:
                self.tokenizer_type = 'bpe'
            
            print(f"Model loaded successfully from {self.model_dir}")
            print(f"Tokenizer type: {self.tokenizer_type}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_input(self, text):
        text = text.strip()
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,!?;:()-]', '', text)
        
        # Add appropriate punctuation if missing
        if not text.endswith(('?', '.', '!', ':', ';')):
            if any(word in text.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']):
                text += '?'
            else:
                text += '.'
        
        return text
    
    def postprocess_output(self, text, input_text):
        text = text.strip()
        
        # Remove input text from beginning if present
        if text.lower().startswith(input_text.lower()):
            text = text[len(input_text):].strip()
        
        # Clean special tokens based on tokenizer type
        if self.tokenizer_type == 'bpe':
            special_tokens = ['<UNK>', '<PAD>', '<BOS>', '<EOS>']
        else:  # sentencepiece
            special_tokens = ['<unk>', '<pad>', '<s>', '</s>', '▁']
        
        for token in special_tokens:
            text = text.replace(token, '')
        
        # Handle SentencePiece specific cleaning
        if self.tokenizer_type == 'sentencepiece':
            text = text.replace('▁', ' ')
            text = re.sub(r'\s+', ' ', text)
        
        # Extract meaningful sentences
        sentences = re.split(r'[.!?]+', text)
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 3 and not sentence.isdigit():
                meaningful_sentences.append(sentence)
        
        if meaningful_sentences:
            # Take first 1-2 meaningful sentences
            if len(meaningful_sentences) >= 2:
                text = '. '.join(meaningful_sentences[:2]) + '.'
            else:
                text = meaningful_sentences[0] + '.'
        else:
            text = "I need more training data to provide a better response."
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\.+', '.', text)
        
        return text
    
    def encode_with_fallback(self, text, max_length=None):
        """Enhanced encoding with fallback handling"""
        try:
            if self.tokenizer_type == 'sentencepiece':
                tokens = self.tokenizer.encode(text)
                if max_length and len(tokens) > max_length - 1:
                    tokens = tokens[:max_length-1]
                    tokens.append(3)  # EOS token for SentencePiece
            else:  # BPE
                tokens = self.tokenizer.encode(text, max_length)
            
            return tokens
        except Exception as e:
            print(f"Encoding error: {e}")
            # Fallback to simple character-level encoding
            return [ord(c) % 1000 for c in text[:max_length or 100]]
    
    def decode_with_fallback(self, token_ids):
        """Enhanced decoding with fallback handling"""
        try:
            if self.tokenizer_type == 'sentencepiece':
                # Filter out invalid token IDs
                valid_tokens = [t for t in token_ids if 0 <= t < self.tokenizer.sp.get_piece_size()]
                return self.tokenizer.decode(valid_tokens)
            else:  # BPE
                return self.tokenizer.decode(token_ids)
        except Exception as e:
            print(f"Decoding error: {e}")
            # Fallback decoding
            return ''.join(chr(min(max(t, 32), 126)) for t in token_ids if t > 0)
    
    def generate_response(self, input_text, max_length=100, temperature=0.8, top_k=50, top_p=0.9):
        processed_input = self.preprocess_input(input_text)
        input_tokens = self.encode_with_fallback(processed_input)
        
        if not input_tokens:
            return "I couldn't process your input. Please try rephrasing."
        
        self.model.eval()
        generated_tokens = input_tokens.copy()
        
        # Get special token IDs based on tokenizer type
        if self.tokenizer_type == 'sentencepiece':
            eos_token = 3
            pad_token = 0
            unk_token = 1
        else:  # BPE
            eos_token = self.tokenizer.vocab.get('<EOS>', -1)
            pad_token = self.tokenizer.vocab.get('<PAD>', -1)
            unk_token = self.tokenizer.vocab.get('<UNK>', -1)
        
        with torch.no_grad():
            for step in range(max_length):
                if len(generated_tokens) >= self.model.max_seq_len:
                    break
                
                # Prepare input tensor
                current_input = torch.tensor([generated_tokens[-self.model.max_seq_len:]], 
                                           dtype=torch.long, device=self.device)
                
                try:
                    logits, _ = self.model(current_input)
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                    
                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    
                    # Check for stopping conditions
                    if next_token == eos_token or next_token == pad_token:
                        break
                    
                    # Skip unknown tokens in short responses
                    if next_token == unk_token and len(generated_tokens) < len(input_tokens) + 5:
                        continue
                    
                    generated_tokens.append(next_token)
                    
                    # Check for natural stopping points
                    if step > 10:  # Only check after generating some tokens
                        recent_text = self.decode_with_fallback(generated_tokens[-20:])
                        if any(punct in recent_text[-5:] for punct in ['.', '!', '?']) and len(generated_tokens) > len(input_tokens) + 10:
                            break
                
                except Exception as e:
                    print(f"Generation error at step {step}: {e}")
                    break
        
        # Decode and postprocess
        full_response = self.decode_with_fallback(generated_tokens)
        response = self.postprocess_output(full_response, processed_input)
        
        return response
    
    def generate_text(self, prompt, max_length=150, temperature=0.9, top_k=40, top_p=0.8):
        prompt_tokens = self.encode_with_fallback(prompt)
        
        if not prompt_tokens:
            return "Unable to process the prompt."
        
        self.model.eval()
        generated_tokens = prompt_tokens.copy()
        
        # Get special token IDs
        if self.tokenizer_type == 'sentencepiece':
            eos_token = 3
            pad_token = 0
        else:
            eos_token = self.tokenizer.vocab.get('<EOS>', -1)
            pad_token = self.tokenizer.vocab.get('<PAD>', -1)
        
        with torch.no_grad():
            for step in range(max_length):
                if len(generated_tokens) >= self.model.max_seq_len:
                    break
                
                current_input = torch.tensor([generated_tokens[-self.model.max_seq_len:]], 
                                           dtype=torch.long, device=self.device)
                
                try:
                    logits, _ = self.model(current_input)
                    next_token_logits = logits[0, -1, :] / temperature
                    
                    # Apply top-k and top-p filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(0, top_k_indices, top_k_logits)
                    
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    
                    if next_token == eos_token:
                        break
                    if next_token == pad_token:
                        continue
                    
                    generated_tokens.append(next_token)
                
                except Exception as e:
                    print(f"Text generation error at step {step}: {e}")
                    break
        
        # Decode and clean up
        generated_text = self.decode_with_fallback(generated_tokens)
        
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # Clean special tokens
        if self.tokenizer_type == 'bpe':
            special_tokens = ['<UNK>', '<PAD>', '<BOS>', '<EOS>']
        else:
            special_tokens = ['<unk>', '<pad>', '<s>', '</s>']
        
        for token in special_tokens:
            generated_text = generated_text.replace(token, '')
        
        if self.tokenizer_type == 'sentencepiece':
            generated_text = generated_text.replace('▁', ' ')
        
        generated_text = re.sub(r'\s+', ' ', generated_text).strip()
        
        return generated_text if generated_text else "Unable to generate meaningful text."
    
    def chat_response(self, user_input, context_length=3):
        self.conversation_history.append(f"Human: {user_input}")
        
        if len(self.conversation_history) > context_length * 2:
            self.conversation_history = self.conversation_history[-context_length * 2:]
        
        context = " ".join(self.conversation_history[-context_length:])
        full_prompt = f"{context} AI:"
        
        response = self.generate_response(full_prompt, max_length=80, temperature=0.7)
        
        self.conversation_history.append(f"AI: {response}")
        
        return response
    
    def get_model_info(self):
        if self.model is None:
            return {"error": "No model loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        vocab_size = None
        if self.tokenizer_type == 'sentencepiece':
            vocab_size = self.tokenizer.sp.get_piece_size() if self.tokenizer.sp else self.tokenizer.vocab_size
        else:
            vocab_size = len(self.tokenizer.vocab) if hasattr(self.tokenizer, 'vocab') else 'Unknown'
        
        return {
            "Model Parameters": f"{total_params:,}",
            "Trainable Parameters": f"{trainable_params:,}",
            "Vocabulary Size": vocab_size,
            "Tokenizer Type": self.tokenizer_type,
            "Model Dimension": self.model.d_model,
            "Number of Layers": len(self.model.transformer_blocks),
            "Max Sequence Length": self.model.max_seq_len,
            "Device": str(self.device)
        }
    
    def clear_conversation(self):
        self.conversation_history = []
    
    def save_conversation(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for line in self.conversation_history:
                f.write(line + '\n')
    
    def benchmark_generation(self, test_prompts, max_length=50):
        results = []
        
        for prompt in test_prompts:
            start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
            
            if self.device.type == 'cuda':
                start_time.record()
            
            response = self.generate_response(prompt, max_length=max_length)
            
            if self.device.type == 'cuda':
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time) / 1000.0
            else:
                elapsed_time = 0.0
            
            response_tokens = len(self.encode_with_fallback(response))
            
            results.append({
                'prompt': prompt,
                'response': response,
                'time_seconds': elapsed_time,
                'tokens_generated': response_tokens,
                'tokens_per_second': response_tokens / elapsed_time if elapsed_time > 0 else 0
            })
        
        return results
    
    def analyze_tokenization(self, text):
        """Analyze how text is tokenized"""
        tokens = self.encode_with_fallback(text)
        decoded = self.decode_with_fallback(tokens)
        
        analysis = {
            'original_text': text,
            'token_count': len(tokens),
            'tokens': tokens,
            'decoded_text': decoded,
            'tokenizer_type': self.tokenizer_type,
            'compression_ratio': len(text) / len(tokens) if tokens else 0
        }
        
        if self.tokenizer_type == 'sentencepiece' and self.tokenizer.sp:
            pieces = [self.tokenizer.sp.id_to_piece(token_id) for token_id in tokens]
            analysis['pieces'] = pieces
        
        return analysis