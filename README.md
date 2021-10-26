# citation-parser
A deep learning model using Transformer and CRF to solve citation parsing problem.

Dataset Used: [GIANT](https://github.com/BeelGroup/GIANT-The-1-Billion-Annotated-Synthetic-Bibliographic-Reference-String-Dataset)

### Model Architecture:
- Feature: Word Embedding & Character Embedding(from BiLSTM)
- Features are fed into transformer encoder layer which got attention module
- Outputs from encoder layer are passed through 2-layers of fully connected linear layer
- Output of linear layer is fed into CRF to find out the most possible tag

### Tokenization Details
Built custom tokenizer which gives individual tag to each punctuation and label each word according to annotation on the citation string.

BIO tagging scheme is used for the labels.
