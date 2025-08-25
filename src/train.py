model = TransformerTranslator()
model.to(device)
optimizer = AdamW(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.001
)
criterion = nn.CrossEntropyLoss(
    ignore_index=sr_tokenizer.pad_token_id,
    reduction='mean')

def train_model(w_dataloader, all_dataloader, p_dataloader, val_loader=None, epoch_len=150):
    model.to(device)
    patience_counter = 0
    model.train()
    scaler = GradScaler()  # For mixed precision
    best_loss = float('inf')
    patience, patience_counter = 3, 0  # Early stopping

    total_steps = (len(all_dataloader) // 2) * epoch_len
    warmup_steps = int(0.1 * total_steps)  # 10% warmup
    best_val_loss = float('inf')

    # Initialize scheduler ONCE before training loop
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    for epoch in range(epoch_len):
        epoch_loss = 0.0
        torch.cuda.empty_cache()  # Force cleanup before epoch
        print(f"\nGPU Memory Start: {torch.cuda.memory_allocated()/1e9:.2f}GB")

        # Dynamic dataloader switching (keep your logic)
        if epoch < epoch_len * 0.05:
            dataloader = w_dataloader
        elif epoch < epoch_len * 0.15:
            dataloader = p_dataloader
        else:
            dataloader = all_dataloader

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            src_key_padding_mask = (batch["src_attention_mask"] == 0).to(device)
            tgt_key_padding_mask = (batch["tgt_attention_mask"][:, :-1] == 0).to(device)

            # Mixed Precision (FP16/BF16)
            with autocast(dtype=torch.float16, device_type=device, enabled=True):  # Use torch.bfloat16 if A100
                outputs = model(
                    src=batch["input_ids"],
                    tgt=batch["labels"][:, :-1],
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask
                )
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    batch["labels"][:, 1:].reshape(-1)
                )

            # Gradient Scaling + Accumulation (2 steps)
            scaler.scale(loss).backward()
            if (i + 1) % 2 == 0 or i == len(dataloader) - 1:  # Adjust accumulation steps as needed
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                epoch_loss += loss.item()
                avg_train_loss = epoch_loss / (len(dataloader) // 2)

                if i % 10 == 0:
                  current_lr = scheduler.get_last_lr()[0]
                  print(f"Epoch {epoch:03d} | Batch {i:03d} | "
                  f"LR {current_lr:.2e} | Train Loss {loss.item():.4f}")

                total_norm = 0
                for p in model.parameters():
                  if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                if total_norm > 5:
                  print(f"Grad norm: {total_norm}")
                print("-------------------------------------")
                print(en_tokenizer.decode(batch['input_ids'][0]))
                print(sr_tokenizer.decode(batch['labels'][0]))
                print("-------------------------------------")


        # --- Validation Phase ---
        if val_loader is not None:
          model.eval()
          val_loss = 0.0
          with torch.inference_mode():
            with autocast(dtype=torch.float16, device_type=device):
              for batch in val_loader:
                  src_key_padding_mask = (batch["src_attention_mask"] == 0).to(device)
                  tgt_key_padding_mask = (batch["tgt_attention_mask"][:, :-1] == 0).to(device)
                  outputs = model(batch["input_ids"], batch["labels"][:, :-1], src_key_padding_mask, tgt_key_padding_mask)
                  loss = criterion(outputs.view(-1, outputs.size(-1)),
                                batch["labels"][:, 1:].reshape(-1))
                  val_loss += loss.item()

          #avg_val_loss = val_loss / len(val_loader)
          model.train()
        avg_val_loss = 0

        if epoch == int(epoch_len * 0.05) or epoch == int(epoch_len * 0.15):
          print(f"Switched to dataloader with batch size: {dataloader.batch_size}")

        # --- Epoch Logging ---
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n[Epoch {epoch + 1:03d}/{epoch_len:03d}] "
              f"LR = {current_lr:.2e} | "
              f"Train Loss = {avg_train_loss:.4f} | "
              f"Val Loss = {avg_val_loss:.4f} | "
              f"Δ Loss = {avg_val_loss - avg_train_loss:+.4f}\n")