# Dhamm_AI_Chatbot

<h2>Key Code Improvements I Made to CiviBot</h2>
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

<h2>Difference between (app3.py and app4.py)</h2>
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

<h2>Improvements in app5.py - Bloom's Taxonomy</h2>

Cognitive Level Detection: The code now analyzes user questions to identify which of the six Bloom's Taxonomy levels they're operating at (Remember, Understand, Apply, Analyze, Evaluate, Create) based on specific verbs and question structures.
Dynamic Prompt Generation: For each question, the bot generates a customized prompt template tailored to the specific cognitive level, helping the AI provide more appropriate responses.
UI Enhancements:

Added a Bloom's Taxonomy guide in the sidebar
Created an expandable tutorial section for users to learn how to ask questions at different cognitive levels
Added cognitive level identification labels to each answer


Improved Response Quality: The AI now responds differently based on whether a student needs simple recall of facts, deeper understanding, application assistance, analysis guidance, evaluation support, or creative problem-solving help
