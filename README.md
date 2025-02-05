# Chat With Your PDF: Multimodal Data Ingestion

## Overview

This project is a multimodal RAG system that ingests data containing images, text, and tables from the well known paper “Attention Is All You Need”. The system should be able to retrieve relevant information and reason over the multimodal data to answer user queries.

## Technical Solution Design

1. Technical Solution Design (expected output: technical design document or PPT)
- Design an end-to-end architecture for the multimodal data ingestion RAG, using diagrams or flowcharts for clarity.
- Specify the frameworks and tools you would use and justify your choices with comparisons for each step in the RAG design (data ingestion, data retrieval and reasoning).


2. Implementation & Practical Application (expected output: demo, code & evaluation)
- Read and extract text, images, and tables from the paper attached.
- Ensure the extracted data is structured in a way that preserves the relationships between text, images, and tables (e.g., captions for images, references to tables in the text).
- Develop a RAG system (data ingestion, data retrieval and reasoning).
- Implement a retrieval mechanism that can answer the below queries:
    - How is the scaled dot product attention calculated?
    - What is the BLEU score of the model in English to German translation EN-DE?
    - How long were the base and big models trained?
    - Which optimizer was used when training the models?
    - Show me a picture that shows the difference between Scaled Dot-Product Attention and Multi-Head Attention.

- For every different experimentation, evaluate your model's output for the 5 questions above. (Mention the evaluation criteria used)


3. Model Enhancements (expected output : document or PPT)
- Mention the challenges in the current system and how can Agentic RAG can solve it.
