import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim.adamw

#checking GPU Availability
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 8
batch_size = 4

#Data Reading.
# curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

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

# print(encoder('hii there'))
# print(decoder(encoder('hii there')))

data = torch.tensor(encoder(text), dtype=torch.long)
size = int(0.9*len(data))
train_data = data[:size]
val_data = data[size:]

#data loading
def get_data(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data) - block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])

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

#eXperimental pre-trained Transformer (APT)
class XPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets = None):
        logits = self.embed(idx)

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
            logits, loss = self(idx)
            #last time step
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            #sample from the distribution
            nxt_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, nxt_idx), dim=1)
        return idx

model = XPT(vocab_size)
llm = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-2)

for iter in range(1000):
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