# Chat With Your PDF: Multimodal Data Ingestion

## Overview

This project is a multimodal RAG system that ingests data containing images, text, and tables from the well-known paper “Attention Is All You Need”. The system should be able to retrieve relevant information and reason over the multimodal data to answer user queries.

## Technical Solution Design

1. Technical Solution Design (expected output: technical design document or PPT)
   - Design an end-to-end architecture for the multimodal data ingestion RAG, using diagrams or flowcharts for clarity.
   - Specify the frameworks and tools you would use and justify your choices with comparisons for each step in the RAG design (data ingestion, data retrieval, and reasoning).

2. Implementation & Practical Application (expected output: demo, code & evaluation)
   - Read and extract text, images, and tables from the paper attached.
   - Ensure the extracted data is structured in a way that preserves the relationships between text, images, and tables (e.g., captions for images, references to tables in the text).
   - Develop a RAG system (data ingestion, data retrieval, and reasoning).
   - Implement a retrieval mechanism that can answer the below queries:
     - How is the scaled dot product attention calculated?
     - What is the BLEU score of the model in English to German translation EN-DE?
     - How long were the base and big models trained?
     - Which optimizer was used when training the models?
     - Show me a picture that shows the difference between Scaled Dot-Product Attention and Multi-Head Attention.
   - For every different experimentation, evaluate your model's output for the 5 questions above. (Mention the evaluation criteria used)

3. Model Enhancements (expected output: document or PPT)
   - Mention the challenges in the current system and how Agentic RAG can solve it.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chat-with-your-pdf.git
   cd chat-with-your-pdf
   ```

2. Install the dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Set up environment variables by creating a `.env` file in the root directory and adding necessary configurations.

## Usage

To run the system, execute the main script:
```bash
poetry run python src/main.py
```

Follow the on-screen instructions to interact with the system.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please contact Omar Elsherif at omarelsherif010@gmail.com.
