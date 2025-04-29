# Dhamm_AI_Chatbot

Key Code Improvements I Made to CiviBot
(app3.py)
Model & API Integration

Upgraded LLM: Switched from Gemini 1.5 Pro to Llama 3 70B via Groq for better response quality
Multiple API support: Added Groq API alongside Google API for improved reliability

Sentiment Analysis Enhancement

Expanded detection: Advanced from simple confusion detection to multi-sentiment analysis (confused, frustrated, curious, neutral)
Contextual awareness: Added consideration of chat history and question patterns

Memory & Retrieval System

Session persistence: Implemented robust memory management through session state variables
Error recovery: Added mechanisms to handle memory errors and recreate conversation chains
Vector database: Upgraded from FAISS to Chroma with detailed chunk retrieval

Code Quality

Error handling: Added comprehensive error checks throughout the codebase
Session management: Better initialization of session state variables
Modular design: Implemented specialized functions for different responsibilities
Documentation: Added detailed comments explaining functionality

Difference between (app3.py and app4.py)
Enhanced Sentiment Analysis

More sophisticated detection: The second version has significantly improved sentiment analysis with:

Expanded keyword lists: Added more terms for each emotional state
Five sentiment categories: Now includes "appreciative" alongside confused, frustrated, curious, and neutral
Scoring system: Uses a point-based system rather than binary detection
Punctuation analysis: Counts question marks and exclamation points as sentiment indicators
Better context awareness: Analyzes patterns across multiple messages



Improved Prompt Engineering

More specific response guidelines: Tailored instructions for each sentiment type
Concrete follow-up suggestions: Specific questions to ask for each emotional state
Better formatting guidance: Clearer instructions on response structure and length
Chat history integration: Explicit inclusion of chat history in prompt template
