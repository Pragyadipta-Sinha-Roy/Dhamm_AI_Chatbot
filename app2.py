import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key is missing! Set it as an environment variable.")
    st.stop()

# Path to the cleaned transcript file
TRANSCRIPT_FILE = "cleaned_transcript.txt"

# System prompt with sentiment placeholder
SYSTEM_PROMPT = """
You are CiviBot, a helpful and knowledgeable assistant specializing in civil engineering concepts. Your primary goal is to help students understand their lecture material by providing clear, accurate explanations about civil engineering topics.

## Your Knowledge Base
- You have access to a repository of civil engineering lecture transcripts.
- You can retrieve relevant information from these transcripts to answer questions.
- If asked about something outside your knowledge base, acknowledge the limitations and offer to help with what you do know.

## Your Personality
- Friendly and approachable - use conversational language that's easy to understand
- Patient - students may be confused or frustrated when they come to you
- Educational - focus on explaining concepts clearly, using examples when helpful
- Encouraging - boost students' confidence in their ability to understand difficult concepts

## User Sentiment Considerations
The user's sentiment has been analyzed and identified as: {sentiment}

Based on this sentiment:
- **If positive/neutral**: Maintain your friendly, informative tone and focus on direct answers.
- **If confused/uncertain**: Use simpler language, break down concepts into smaller parts, and offer examples. Ask if they'd like further clarification.
- **If frustrated**: Acknowledge their difficulty, be extra supportive, and offer multiple approaches to understanding the concept. Reassure them that many students find this challenging.
- **If curious/excited**: Match their enthusiasm, provide additional interesting details, and suggest related topics they might find interesting.

## Response Guidelines
- Keep explanations concise but complete
- Use bullet points for lists of steps or related concepts
- Format mathematical equations clearly when needed
- Refer to specific sections of lectures when relevant
- Always end with an offer to help further if the explanation wasn't sufficient

Remember: Your goal is to help students understand civil engineering concepts, not just provide information. Success means they leave the conversation with greater clarity and confidence.

Now, using the context provided and considering the user's sentiment, please answer this question: {question}
"""

def load_transcript():
    if not os.path.exists(TRANSCRIPT_FILE):
        st.error(f"Transcript file '{TRANSCRIPT_FILE}' not found!")
        return ""
    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        api_key=GOOGLE_API_KEY
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def detect_sentiment(text, chat_history=None):
   
    confusion_keywords = ["confused", "not sure", "don't get", "difficult", "unclear", "what", "hard", 
                         "don't understand", "explain", "how does", "what is", "?"]
    frustration_keywords = ["frustrated", "annoying", "still don't get", "not making sense", 
                           "too difficult", "impossible", "giving up", "waste", "useless", "!"]
    curiosity_keywords = ["interesting", "cool", "awesome", "fascinating", "tell me more", 
                         "curious", "excited", "wonder", "how about"]
    
    text_lower = text.lower()
    
    if any(kw in text_lower for kw in frustration_keywords):
        return "frustrated"
    
    elif any(kw in text_lower for kw in confusion_keywords) or text_lower.count("?") > 1:
        return "confused"
    
    elif any(kw in text_lower for kw in curiosity_keywords):
        return "curious"
    
    elif chat_history and len(chat_history) > 2:
        last_user_msg = chat_history[-2].content.lower()
        
        if any(kw in last_user_msg for kw in confusion_keywords) and len(text_lower.split()) < 8:
            return "confused"
    
    return "neutral"

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-pro-latest', 
        api_key=GOOGLE_API_KEY
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True
    )

def handle_userinput(question):
    with st.spinner("Thinking..."):
        # Detect sentiment from current question and chat history
        sentiment = detect_sentiment(
            question, 
            st.session_state.chat_history if st.session_state.chat_history else None
        )
        
        # Store the detected sentiment for internal use
        st.session_state.current_sentiment = sentiment
        
        # Special system instruction with sentiment context
        system_instruction = f"The user's detected sentiment is: {sentiment}. Keep this in mind while responding. "
        
        # Get response with the special instruction prepended
        response = st.session_state.conversation({
            "question": system_instruction + question
        })
        
        st.session_state.chat_history = response['chat_history']
    
    # Add the new Q&A pair to our persistent display list
    st.session_state.qa_pairs.append({
        "question": question,
        "answer": response['answer']
    })

def display_qa_history():
    """Display all Q&A pairs in the session history"""
    for i, qa in enumerate(st.session_state.qa_pairs):
        with st.container():
            st.markdown(f"### Question {i+1}")
            st.markdown(f"**Q:** {qa['question']}")
            st.markdown(f"**A:** {qa['answer']}")
            st.markdown("---")

def main():
    st.set_page_config(page_title="CiviBot - Your Civil Engineering Assistant", page_icon=":construction_worker:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "current_sentiment" not in st.session_state:
        st.session_state.current_sentiment = "neutral"
    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = []

    st.title("👷‍♂️ Meet CiviBot - Your Civil Engineering Buddy")

    st.markdown("""
    Hello! I'm **CiviBot**, your friendly Civil Engineering assistant.  
    I can help explain lectures, clear up concepts, or chat about cement, concrete, and construction.  
    Ask me anything you're stuck on!
    """)

    # Admin panel for debugging (can be hidden in production)
    with st.sidebar:
        st.header("Admin Panel")
        if st.button("Clear Conversation History"):
            st.session_state.qa_pairs = []
            st.session_state.chat_history = None
            st.rerun()
    
    if st.session_state.conversation is None:
        with st.spinner("Loading transcript and getting things ready..."):
            raw_text = load_transcript()
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("CiviBot is ready to chat!")

    # Display Q&A history first so it appears above the input box
    if st.session_state.qa_pairs:
        st.markdown("## Previous Questions & Answers")
        display_qa_history()

    # User input section
    user_question = st.text_input("Ask me something from your lecture:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)
        st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

if __name__ == '__main__':
    main()