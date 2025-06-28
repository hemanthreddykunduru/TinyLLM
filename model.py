import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os
import re
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Tuple
import sentencepiece as spm

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.w_o(context)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x

class CustomLLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.lm_head.weight = self.token_embedding.weight
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_causal_mask(self, seq_len: int, device: torch.device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        seq_len = min(seq_len, self.max_seq_len)
        input_ids = input_ids[:, :seq_len]
        
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos_ids)
        x = self.dropout(token_emb + pos_emb)
        
        mask = self.create_causal_mask(seq_len, device)
        
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        x = self.norm_final(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            targets = targets[:, :seq_len]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                 shift_targets.view(-1), ignore_index=-1)
        
        return logits, loss

class BPETokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_freqs = defaultdict(int)
        self.vocab = {}
        self.merges = []
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
    def _get_word_tokens(self, word: str) -> List[str]:
        return list(word) + ['</w>']
    
    def _get_pairs(self, word_tokens: List[str]) -> set:
        pairs = set()
        prev_char = word_tokens[0]
        for char in word_tokens[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def _get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = list(word)
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in vocab:
            new_word = p.sub(''.join(pair), ' '.join(word))
            new_vocab[tuple(new_word.split())] = vocab[word]
        return new_vocab
    
    def train(self, texts: List[str]):
        for text in texts:
            words = text.lower().split()
            for word in words:
                self.word_freqs[word] += 1
        
        vocab = {}
        for word, freq in self.word_freqs.items():
            vocab[tuple(self._get_word_tokens(word))] = freq
        
        num_merges = self.vocab_size - len(self.special_tokens) - len(set(''.join(self.word_freqs.keys())))
        
        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
        
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        
        all_tokens = set()
        for word_tokens in vocab.keys():
            all_tokens.update(word_tokens)
        
        for token in sorted(all_tokens):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
    
    def _apply_merges(self, word_tokens: List[str]) -> List[str]:
        pairs = self._get_pairs(word_tokens)
        
        if not pairs:
            return word_tokens
        
        while True:
            bigram = min(pairs, key=lambda pair: self.merges.index(pair) if pair in self.merges else float('inf'))
            
            if bigram not in self.merges:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word_tokens):
                try:
                    j = word_tokens.index(first, i)
                    new_word.extend(word_tokens[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word_tokens[i:])
                    break
                
                if word_tokens[i] == first and i < len(word_tokens) - 1 and word_tokens[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word_tokens[i])
                    i += 1
            
            word_tokens = new_word
            if len(word_tokens) == 1:
                break
            else:
                pairs = self._get_pairs(word_tokens)
        
        return word_tokens
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        tokens = []
        words = text.lower().split()
        
        for word in words:
            word_tokens = self._get_word_tokens(word)
            word_tokens = self._apply_merges(word_tokens)
            
            for token in word_tokens:
                tokens.append(self.vocab.get(token, self.vocab['<UNK>']))
        
        if max_length:
            tokens = tokens[:max_length-1]
            tokens.append(self.vocab['<EOS>'])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for token_id in token_ids:
            for token, idx in self.vocab.items():
                if idx == token_id:
                    tokens.append(token)
                    break
        
        text = ''.join(tokens).replace('</w>', ' ').strip()
        for special in self.special_tokens:
            text = text.replace(special, '')
        
        return text
    
    def save(self, path: str):
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': len(self.vocab),
            'word_freqs': dict(self.word_freqs)
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.merges = [tuple(merge) for merge in data['merges']]
            self.word_freqs = defaultdict(int, data['word_freqs'])

class SentencePieceTokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.model_path = None
        self.sp = None
        
    def train(self, texts: List[str], model_path: str):
        self.model_path = model_path
        
        with open(f"{model_path}.txt", 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        spm.SentencePieceTrainer.train(
            input=f"{model_path}.txt",
            model_prefix=model_path,
            vocab_size=self.vocab_size,
            character_coverage=0.9995,
            model_type='bpe',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<PAD>',
            unk_piece='<UNK>',
            bos_piece='<BOS>',
            eos_piece='<EOS>'
        )
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{model_path}.model")
        
        os.remove(f"{model_path}.txt")
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        if self.sp is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        tokens = self.sp.encode_as_ids(text)
        
        if max_length:
            tokens = tokens[:max_length-1]
            tokens.append(3)  # EOS token
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        if self.sp is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        return self.sp.decode_ids(token_ids)
    
    def save(self, path: str):
        if self.model_path is None:
            raise ValueError("No model to save")
        
        import shutil
        shutil.copy(f"{self.model_path}.model", f"{path}.model")
        shutil.copy(f"{self.model_path}.vocab", f"{path}.vocab")
        
        with open(f"{path}_config.json", 'w') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'model_path': path
            }, f)
    
    def load(self, path: str):
        with open(f"{path}_config.json", 'r') as f:
            config = json.load(f)
            self.vocab_size = config['vocab_size']
            self.model_path = config['model_path']
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f"{path}.model")
    
    @property
    def vocab_size_actual(self):
        return self.sp.get_piece_size() if self.sp else self.vocab_size

def create_tokenizer(tokenizer_type: str = "bpe", vocab_size: int = 10000):
    if tokenizer_type.lower() == "bpe":
        return BPETokenizer(vocab_size)
    elif tokenizer_type.lower() == "sentencepiece":
        return SentencePieceTokenizer(vocab_size)
    else:
        raise ValueError("tokenizer_type must be 'bpe' or 'sentencepiece'")

def save_model(model, tokenizer, save_dir, optimizer=None, epoch=None, loss=None):
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
    
    config = {
        'vocab_size': model.vocab_size,
        'd_model': model.d_model,
        'n_heads': model.n_heads,
        'n_layers': model.n_layers,
        'd_ff': model.d_ff,
        'max_seq_len': model.max_seq_len,
        'dropout': model.dropout_rate
    }
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    if tokenizer:
        if hasattr(tokenizer, 'sp') and tokenizer.sp is not None:
            tokenizer.save(os.path.join(save_dir, 'tokenizer'))
            with open(os.path.join(save_dir, 'tokenizer_type.txt'), 'w') as f:
                f.write('sentencepiece')
        else:
            tokenizer.save(os.path.join(save_dir, 'tokenizer.json'))
            with open(os.path.join(save_dir, 'tokenizer_type.txt'), 'w') as f:
                f.write('bpe')
    
    if optimizer and epoch is not None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        torch.save(checkpoint, os.path.join(save_dir, 'checkpoint.pth'))

def load_model(model_dir, device):
    with open(os.path.join(model_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    print(f"Loading model with config: {config}")
    
    model = CustomLLM(**config)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pth'), 
                                   map_location=device))
    model.to(device)
    model.eval()
    
    with open(os.path.join(model_dir, 'tokenizer_type.txt'), 'r') as f:
        tokenizer_type = f.read().strip()
    
    if tokenizer_type == 'sentencepiece':
        tokenizer = SentencePieceTokenizer()
        tokenizer.load(os.path.join(model_dir, 'tokenizer'))
    else:
        tokenizer = BPETokenizer()
        tokenizer.load(os.path.join(model_dir, 'tokenizer.json'))
    
    return model, tokenizer