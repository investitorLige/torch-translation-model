# Legal Document Translator (English → Serbian)

A PyTorch-based Transformer model for translating legal documents from English to Serbian.
The model is designed to handle complex legal language, preserve terminology, and maintain document structure where possible.

# Features

Transformer architecture built with PyTorch.

Custom tokenization using BERT (English) and BCMS-BERTić (Serbian).

Mixed precision training with dynamic learning rate scheduling.

# Roadmap

 Add support for Serbian Cyrillic ↔ Latin script conversion.

 Train own joint tokenizer for both English and Serbian

 Improve handling of named entities (laws, institutions).


# Contributing

Contributions are welcome!

Fork the repo

Create a feature branch (git checkout -b feature/new-feature)

Commit your changes (git commit -m 'Add new feature')

Push to your branch (git push origin feature/new-feature)

Open a Pull Request

# License

Distributed under the MIT License. See LICENSE for details.

# Acknowledgements

PyTorch

HuggingFace Transformers

classla/bcms-bertic
 for Serbian NLP

Inspiration from OpenNMT and academic work on legal translation

# Important Notes
The src file is only for reference, everything is done in colab notebooks using their gps from the cloud.
TODO: organize the notebook better.

The src file is working fine but no model is trained and there are no datasets nor data.

(Optional) Enable mixed precision with NVIDIA Apex:
!pip install nvidia-apex
