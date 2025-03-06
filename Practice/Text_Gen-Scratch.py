import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim.adamw

BLOCK_SIZE = 256
BATCH_SIZE = 64
NUM_EMBEDDINGS = 384
NUM_HEADS = 6
NUM_LAYERS = 6
#Data Reading.
# curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

#checking GPU Availability
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def tokenizers(data):
    ch_i = {}
    i_ch = {}
    for i, ch in enumerate(data):
        ch_i[ch] = i
        i_ch[i] = ch
    return ch_i, i_ch

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

chars =sorted(list(set(text)))
vocab_size = len(chars)
ch_i, i_ch = tokenizers(chars)

encoder = lambda e : [ch_i[c] for c in e]
decoder = lambda d : ''.join([i_ch[i] for i in d])

data = torch.tensor(encoder(text), dtype=torch.long)
size = int(0.9*len(data))
train_data = data[:size]
val_data = data[size:]

#data loading
def get_data(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - BLOCK_SIZE,(BATCH_SIZE,))
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in idx])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in idx])

    x , y = x.to(device), y.to(device)

    return x,y

@torch.no_grad()
def evaluate_loss():
    opt = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X,Y = get_data(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        opt[str(split)] = losses.mean()
    model.train()
    return opt

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.NN(x)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(NUM_EMBEDDINGS, head_size, bias=False)
        self.query = nn.Linear(NUM_EMBEDDINGS, head_size, bias=False)
        self.value = nn.Linear(NUM_EMBEDDINGS, head_size, bias=False)
        self.register_buffer('trill',torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout()
    
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        output = wei @ v
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.ln = nn.Linear(head_size * n_heads, NUM_EMBEDDINGS)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(out)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

#eXperimental pre-trained Transformer (XPT)
class XPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, NUM_EMBEDDINGS)
        self.pos_embed = nn.Embedding(BLOCK_SIZE, NUM_EMBEDDINGS)
        self.block = nn.Sequential(*[Block(NUM_EMBEDDINGS, NUM_HEADS) for _ in range(NUM_LAYERS)])
        self.lm_head = nn.Linear(NUM_EMBEDDINGS, vocab_size)
        self.ln_layer = nn.LayerNorm(NUM_EMBEDDINGS)
    
    def forward(self, idx, targets = None):
        
        B,T = idx.shape
        token_embd = self.embed(idx)
        pos_embd = self.pos_embed(torch.arange(T, device=device))
        x = token_embd + pos_embd
        x = self.block(x)
        x = self.ln_layer(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss =  F.cross_entropy(logits, targets)
        return logits,loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            ix = idx[:, -BLOCK_SIZE:]
            logits, loss = self(ix)
            #last time step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            #sample from the distribution
            nxt_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt_idx), dim=1)
        return idx

model = XPT()
llm = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for iter in range(5000):
    if iter%100 == 0:

        losses = evaluate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_data('train')
    logits, loss = model(xb,yb)
    optimizer.zero_grad(set_to_none =True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decoder(llm.generate(context, max_new_tokens=500)[0].tolist()))
