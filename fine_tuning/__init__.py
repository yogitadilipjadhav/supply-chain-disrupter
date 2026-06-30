"""
fine_tuning package — offline model training workflows for Capstone Project 8.

Modules:
  generate_training_data  — SQLite/ChromaDB → training splits and QA pairs
  finetune_distilbert     — Signal 2: 4-class DistilBERT risk classifier
  finetune_embeddings     — RAG bi-encoder (all-MiniLM-L6-v2)
  finetune_gpt4o_mini     — News Agent GPT-4o-mini (OpenAI API, optional)
  evaluate_all            — Day 23 consolidated evaluation report
  colab_finetune_and_upload — One-shot Colab GPU pipeline + Hugging Face upload
"""
