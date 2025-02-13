# PDF Extraction Results
- Source: attention_paper.pdf
- Total PDF Pages: 15
- Extracted Pages: 1

---

Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.

# Attention Is All You Need

arXiv:1706.03762v7  [cs.CL]  2 Aug 2023

Ashish Vaswani∗

Noam Shazeer∗

Niki Parmar∗

Jakob Uszkoreit∗

Google Brain

Google Brain

Google Research

Google Research

avaswani@google.com

noam@google.com

nikip@google.com

usz@google.com

Llion Jones∗

Aidan N. Gomez∗ †

Łukasz Kaiser∗

Google Research

University of Toronto

Google Brain

llion@google.com

aidan@cs.toronto.edu

lukaszkaiser@google.com

Illia Polosukhin∗ ‡

illia.polosukhin@gmail.com

# Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

∗ Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.

‡Work performed while at Google Research.

†Work performed while at Google Brain.

31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.