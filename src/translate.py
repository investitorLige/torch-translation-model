def translate_single(
    model,
    english_text,
    en_tokenizer,
    sr_tokenizer,
    max_length=64,
    beam_width=5,
    length_penalty=0.6,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Tokenize input with [CLS]/[SEP]
    inputs = en_tokenizer(
        english_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
        add_special_tokens=True  # Ensure [CLS] and [SEP] are added
    ).to(device)
    model.to(device)

    # Initialize beams with [CLS] token
    cls_id = sr_tokenizer.cls_token_id
    sep_id = sr_tokenizer.sep_token_id
    assert cls_id is not None, "Tokenizer missing [CLS] token"
    assert sep_id is not None, "Tokenizer missing [SEP] token"
    beams = [([cls_id], 0.0)]  # Start with [CLS]

    model.eval()
    with torch.inference_mode():
        # Forward pass through encoder
        src_emb = (model.pos_encoder(model.src_embedding(inputs["input_ids"])) * math.sqrt(model.d_model)).to(device)
        memory = model.transformer.encoder(
            src_emb,
            src_key_padding_mask=(inputs["attention_mask"] == 0)
        ).to(device)

        for _ in range(max_length):
            candidates = []
            for seq, score in beams:
                # Stop if [SEP] is generated
                if seq[-1] == sep_id:
                    candidates.append((seq, score))
                    continue

                # Prepare decoder input
                tgt = torch.tensor([seq], device=device)
                tgt_emb = model.pos_encoder(model.tgt_embedding(tgt)) * math.sqrt(model.d_model)

                # Forward pass through decoder
                output = model.transformer.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=model.transformer.generate_square_subsequent_mask(tgt.size(1)).to(device),
                    memory_key_padding_mask=(inputs["attention_mask"] == 0)
                )
                logits = model.fc_out(output[:, -1, :])

                # Get top-k tokens
                topk_scores, topk_tokens = torch.topk(logits, beam_width)
                topk_scores = torch.log_softmax(topk_scores, dim=-1)

                for i in range(beam_width):
                    new_seq = seq + [topk_tokens[0, i].item()]
                    new_score = score + topk_scores[0, i].item()
                    candidates.append((new_seq, new_score))

            # Select top beams
            print(f"Step {_}: Top beam -> {sr_tokenizer.decode(beams[-1][0])}")
            beams = sorted(candidates, key=lambda x: x[1])[-beam_width:]
            print(f"Step {_}: Top beam -> {sr_tokenizer.decode(beams[-1][0])}")

            # Early stopping if all beams end with [SEP]
            if all(beam[0][-1] == sep_id for beam in beams):
                break

    # Select best beam (with length penalty)
    if not beams:  # Handle empty beam case
        return ""
    #print(beams)
    print([sr_tokenizer.decode(x) for x in [word[0] for word in beams]])
    best_beam = max(beams, key=lambda x: x[1] / (len(x[0])**length_penalty))
    return sr_tokenizer.decode(best_beam[0], skip_special_tokens=True)

def greedy_translate_single(
    model,
    english_text,
    en_tokenizer,
    sr_tokenizer,
    max_length=64,
    length_penalty=0.6,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
  inputs = en_tokenizer(
    english_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=max_length,
    add_special_tokens=True  # Ensure [CLS] and [SEP] are added
  ).to(device)

  model.to(device)
  cls_id = sr_tokenizer.cls_token_id
  sep_id = sr_tokenizer.sep_token_id
  assert cls_id is not None, "Tokenizer missing [CLS] token"
  assert sep_id is not None, "Tokenizer missing [SEP] token"
  final_seq = [cls_id]

  model.eval()
  with torch.inference_mode():
    src_emb = (model.pos_encoder(model.src_embedding(inputs["input_ids"])) * math.sqrt(model.d_model)).to(device)
    memory = model.transformer.encoder(
        src_emb,
        src_key_padding_mask=(inputs["attention_mask"] == 0)
    ).to(device)

    for _ in range(max_length):

      if final_seq[-1] == sep_id:
        break

      tgt = torch.tensor(final_seq, device=device).unsqueeze(0)
      tgt_emb = model.pos_encoder(model.tgt_embedding(tgt)) * math.sqrt(model.d_model)

      output = model.transformer.decoder(
      tgt_emb,
      memory,
      tgt_mask=model.transformer.generate_square_subsequent_mask(tgt.size(1)).to(device),
      memory_key_padding_mask=(inputs["attention_mask"] == 0)
      )

      logits = model.fc_out(output[:, -1, :])
      topk_scores, topk_tokens = torch.topk(logits, 1)
      next_token = topk_tokens.squeeze(-1)
      final_seq.append(next_token.item())

  return sr_tokenizer.decode(final_seq, skip_special_tokens=True)
