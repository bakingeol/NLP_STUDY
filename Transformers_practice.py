
#%%
from math import sqrt
from transformers import AutoTokenizer
from torch import nn
from transformers import AutoConfig
from transformers import BertModel
import torch
import torch.nn.functional as F

#%%
model_ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
text = 'time flies like an arrow'
#%%
# 텍스트 토큰화
inputs = tokenizer(text, return_tensors = 'pt', add_special_tokens = False)
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
input_embeds = token_emb(inputs.input_ids)
#%%
input_embeds, input_embeds.size()
#%%
query = key = value = input_embeds
dim_k = key.size(-1)
scores = torch.bmm(query, key.transpose(1,2))/sqrt(dim_k)
#%%
dim_k
#%%
weights = F.softmax(scores, dim = -1)
weights.sum(dim = -1)

attn_outputs = torch.bmm(weights, value)
#%%
def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1,2))/sqrt(dim_k)
    weights = F.softmax(scores, dim = -1)
    return torch.bmm(weights, value)
#%%
class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
        )
        return attn_outputs
#%%  
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size # 768
        num_heads = config.num_attention_heads # 12
        head_dim = embed_dim // num_heads # 투영하려는 차원의 크기 64
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim = -1)
        x = self.output_linear(x)
        
        return x
#%%
MultiHeadAttention(config)
#%%    
multihead_attn = MultiHeadAttention(config)                                                                       
attn_output = multihead_attn(input_embeds)
attn_output
#%%
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size) # 768 -> 3072
        # print(config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size) # 3072 -> 768
        # print(config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self,x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
#%% 
feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_outputs)
# %%
ff_outputs.size()
# %%
config
# %%
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
    
    def forward(self, x):
        # 층 정규화를 진행하고 퀴리, 키, 값으로 복사
        hidden_state = self.layer_norm_1(x)
        x = x + self.attention(hidden_state)
        x = x + self.attention(self.layer_norm_2(x))
        return x    
# %%
encoder_layer = TransformerEncoderLayer(config)
input_embeds.shape, encoder_layer(input_embeds).size()
#%%
config
# %% 위치 임베딩
class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size,
                                             config.hidden_size) # 30522, 768
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size) # 512, 768
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps = 1e-12)
        self.dropout = nn.Dropout()
    
    # 'input_ids': tensor([[ 2051, 10029,  2066,  2019,  8612]])
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype = torch.long).unsqueeze(0)
        #print(input_ids.size(1), position_ids)
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings        
# %%
embedding_layer = Embeddings(config)
embedding_layer(inputs.input_ids).size()

#%%
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config)\
                                     for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x
# %%
encoder = TransformerEncoder(config)
encoder(inputs.input_ids).size()
# %% 분류 헤드 추가하기
class TrnasofrmerForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, x):
        x = self.encoder(x)[:,0,:]
        #print(x)
        x = self.dropout(x)
        #print(x)
        x = self.classifier(x)
        return x
        
#%%
config.num_labels = 3
encoder_classifier = TrnasofrmerForSequenceClassification(config)
encoder_classifier(inputs.input_ids).size()
# %%
