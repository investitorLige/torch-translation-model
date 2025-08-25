en_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sr_tokenizer = BertTokenizer.from_pretrained("classla/bcms-bertic")
if sr_tokenizer.pad_token is None:
  sr_tokenizer.add_special_tokens({'pad_token': '<pad>'})
  print("added sr pad")
if en_tokenizer.pad_token is None:
  print("added en pad")
  en_tokenizer.add_special_tokens({'pad_token': '<pad>'})
print(en_tokenizer.pad_token_id)


class TransformerTranslator(nn.Module):
    def __init__(
        self,
        src_vocab_size=32000,  # English vocabulary size
        tgt_vocab_size=32000,  # Serbian vocabulary size
        d_model=1024,           # Embedding dimension
        nhead=16,              # Attention heads
        num_encoder_layers=10,
        num_decoder_layers=8,
        dim_feedforward=3072,  # Hidden layer size
        norm_first=True,
        activation="gelu",
        dropout=0.2,
        use_checkpointing=False  # New parameter to control checkpointing
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing  # Store the flag
        self.d_model = d_model


        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=norm_first
        )

        # Output layer
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = self.pos_encoder(self.src_embedding(src).to(device)) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(self.tgt_embedding(tgt).to(device)) * math.sqrt(self.d_model)

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
        memory_key_padding_mask = src_key_padding_mask
        src_key_padding_mask.to(device)
        if memory_key_padding_mask is not None:
          memory_key_padding_mask.to(device)


        if self.use_checkpointing:
            # Manually run checkpointed transformer
            output = self._checkpointed_transformer_forward(
                src=src,
                tgt=tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        else:
            # print(f"tgt shape: {tgt.shape}")
            # print(f"src shape:  + {src.shape}")
            # print(f"tgt mask shape:  + {tgt_mask.shape}")
            # print(f"src key padding mask shape:  + {src_key_padding_mask.shape}")
            # print(f"tgt key padding mask shape:  + {tgt_key_padding_mask.shape}")
            output = self.transformer(
                src=src,
                tgt=tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        return self.fc_out(output)

    def _checkpointed_transformer_forward(self, src, tgt, tgt_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):
      # Requires PyTorch >= 2.0
      for mod in self.transformer.encoder.layers:
          src = checkpoint(mod, src, use_reentrant=False)
      for mod in self.transformer.decoder.layers:
          tgt = checkpoint(mod, tgt, memory=src, use_reentrant=False)
      return tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].to(device)
        return self.dropout(x)