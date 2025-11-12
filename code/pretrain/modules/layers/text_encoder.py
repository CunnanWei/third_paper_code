from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, 
                 text_model='ncbi/MedCPT-Query-Encoder',
                 free_layers=6,           # 冻结前6层
                 proj_hidden=256,         # 投影层隐藏维度
                 proj_out=256):           # 投影层输出维度
        super().__init__()
        
        # ========== 1. Text Encoder ==========
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model             
        )
        self.lm_model = AutoModel.from_pretrained(
            text_model            
        )
        
        # ========== 2. 冻结前N层==========
        if free_layers is not None:
            for layer_idx in range(int(free_layers)):
                for param in self.lm_model.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False
        
        # ========== 3. Text Projector==========
        self.proj_t = nn.Sequential(
            nn.Linear(768, proj_hidden),      # 768 → 256
            nn.GELU(),                         
            nn.Linear(proj_hidden, proj_out), # 256 → 256
        )
    
    def _tokenize(self, text, device=None):
        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            add_special_tokens=True,    
            truncation=True,             
            max_length=256,              
            padding='max_length',        
            return_tensors='pt'          
        )
        if device is not None:
            tokenizer_output = tokenizer_output.to(device)
        return tokenizer_output
    
    def get_text_emb(self, input_ids, attention_mask):
        text_emb = self.lm_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).pooler_output  
        
        return text_emb
    
    def forward(self, input_ids, mask=None):
        # 传入的 input_ids: LongTensor[b, seq]
        # mask: BoolTensor 或 LongTensor[b, seq]，为 True/1 表示有效 token
        input_ids = input_ids.long()
        if mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        else:
            attention_mask = mask.long()

        # 1. 获取Text Encoder嵌入 [batch, 768]
        text_emb = self.get_text_emb(input_ids, attention_mask)

        # 2. 投影到共享空间 [batch, 256]
        proj_text_emb = self.proj_t(text_emb.contiguous())

        return proj_text_emb

if __name__ == "__main__":
    text_encoder = TextEncoder()
    text = ["窦性心律正常", "房颤伴快室率", "ST 段压低", "T 波倒置"]
    tokenizer_output = text_encoder._tokenize(text)
    input_ids = tokenizer_output.input_ids
    text_emb = text_encoder(input_ids)
    print(text_emb.shape)
