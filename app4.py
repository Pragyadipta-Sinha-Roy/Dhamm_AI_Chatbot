import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from pyprojroot import here

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API Key is missing! Set it as an environment variable.")
    st.stop()

if not GROQ_API_KEY:
    st.error("Groq API Key is missing! Set it as an environment variable.")
    st.stop()

EMBEDDING_MODEL = "models/text-embedding-004"
VECTORDB_DIR = r"C:\Users\KIIT0001\Documents\ML project\Dhamm_AI_Chatbot\vectordb"
COLLECTION_NAME = "chroma"
K = 2 
TRANSCRIPT_FILE = "cleaned_transcript.txt"

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

def get_vectorstore():
    """Get the pre-existing Chroma vector database"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, 
            api_key=GOOGLE_API_KEY
        )
        
        vectordb = Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(here(VECTORDB_DIR)),
            embedding_function=embeddings
        )
        return vectordb
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
        return None

def detect_sentiment(text, chat_history=None):
    """Enhanced sentiment detection with more nuanced categories and context awareness"""
    
    confusion_keywords = [
        "confused", "not sure", "don't get", "difficult", "unclear", "what", "hard", 
        "don't understand", "explain", "how does", "what is", "?", "lost", "unsure",
        "clarify", "help me understand", "doesn't make sense"
    ]
    
    frustration_keywords = [
        "frustrated", "annoying", "still don't get", "not making sense", 
        "too difficult", "impossible", "giving up", "waste", "useless", "!", 
        "stupid", "ridiculous", "confusing", "irritating", "fed up", "tired of",
        "this is not helpful", "doesn't work", "can't", "won't"
    ]
    
    curiosity_keywords = [
        "interesting", "cool", "awesome", "fascinating", "tell me more", 
        "curious", "excited", "wonder", "how about", "learn", "explore",
        "dig deeper", "more details", "examples", "what else", "really"
    ]
    
    appreciation_keywords = [
        "thanks", "thank you", "helpful", "appreciate", "great", "good job",
        "excellent", "perfect", "makes sense", "clear", "get it now", "understand"
    ]
    
    text_lower = text.lower()
    
    question_mark_count = text_lower.count("?")
    
    exclamation_mark_count = text_lower.count("!")
    
    sentiment_scores = {
        "confused": 0,
        "frustrated": 0,
        "curious": 0,
        "appreciative": 0,
        "neutral": 1  
    }
    
    for kw in confusion_keywords:
        if kw in text_lower:
            sentiment_scores["confused"] += 1
    
    for kw in frustration_keywords:
        if kw in text_lower:
            sentiment_scores["frustrated"] += 1
    
    for kw in curiosity_keywords:
        if kw in text_lower:
            sentiment_scores["curious"] += 1
    
    for kw in appreciation_keywords:
        if kw in text_lower:
            sentiment_scores["appreciative"] += 1
    
    sentiment_scores["confused"] += min(question_mark_count, 3)
    sentiment_scores["frustrated"] += min(exclamation_mark_count, 3)
    
    if len(text_lower.split()) < 5 and chat_history and len(chat_history) >= 2:
        last_user_msg = chat_history[-2].content.lower() if hasattr(chat_history[-2], 'content') else str(chat_history[-2]).lower()
        if any(kw in last_user_msg for kw in confusion_keywords):
            sentiment_scores["confused"] += 1
    
    if chat_history and len(chat_history) >= 4:
        recent_messages = chat_history[-4:]
        user_messages = [msg.content if hasattr(msg, 'content') else str(msg) for msg in recent_messages[::2]]
        if all(len(msg.split()) < 8 for msg in user_messages):
            sentiment_scores["frustrated"] += 1
    
    max_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
    
    if max_sentiment[0] != "neutral" and max_sentiment[1] > 1:
        return max_sentiment[0]
    else:
        return "neutral"

def get_custom_prompt():
    """Create an enhanced prompt template with better user engagement"""
    template = """
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
    - **If confused**: Use simpler language, break down concepts step-by-step, and offer concrete examples. End with a specific follow-up question like "Would it help if I explained [specific aspect] in more detail?" or "Which part is still unclear?"
    
    - **If frustrated**: Acknowledge their frustration directly ("I understand this can be challenging"), be extra supportive, and offer a completely different approach to explaining the concept. Ask "Would it help to approach this from a different angle?" or "Would a real-world example make this clearer?"
    
    - **If curious/excited**: Match their enthusiasm, provide additional interesting details or applications of the concept. Ask engaging follow-ups like "Would you like to explore how this concept applies to [related field/application]?"
    
    - **If appreciative**: Build on their understanding by offering the next logical concept or a more advanced application. Ask "Now that you understand this, would you like to learn about [related concept]?"
    
    - **If neutral**: Provide a clear, concise explanation and ask "Does that answer your question, or would you like more details on a specific aspect?"

    ## Response Guidelines
    - Start with a direct answer to their question whenever possible
    - Keep explanations concise (3-4 paragraphs maximum) but complete
    - For complex topics, break down the answer into clearly labeled sections
    - Use bullet points for lists of steps or related concepts
    - Include 1-2 relevant examples that make abstract concepts concrete
    - Format mathematical equations clearly when needed
    - End with ONE specific follow-up question based on their sentiment
    - IMPORTANT: Reference previous conversation context when appropriate

    ## Relevant Context from lecture transcripts:
    {context}
    
    ## Current Question: 
    {question}
    
    ## Previous Conversation Context (use this to maintain continuity):
    {chat_history}
            
    Helpful Response:
    """
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question", "chat_history"],
        partial_variables={"sentiment": st.session_state.get("current_sentiment", "neutral")}
    )

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key=GROQ_API_KEY,
        temperature=0.5,
        max_tokens=2048
    )
    
    if "memory" not in st.session_state or st.session_state.memory is None:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"  
        )
        st.session_state.memory.chat_memory.messages = []
    
    prompt = get_custom_prompt()
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": K}),
        memory=st.session_state.memory,  
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,  
        verbose=True
    )

def lookup_relevant_chunks(query, vectorstore):
    """Retrieve relevant chunks from vector database directly"""
    docs = vectorstore.similarity_search(query, k=K)
    return docs

def handle_userinput(question):
    with st.spinner("Thinking..."):
        if st.session_state.vectorstore:
            retrieved_chunks = lookup_relevant_chunks(question, st.session_state.vectorstore)
            st.session_state.current_chunks = retrieved_chunks
            
            context = "\n\n".join([doc.page_content for doc in retrieved_chunks])
            st.session_state.current_context = context
        
        sentiment = detect_sentiment(
            question, 
            st.session_state.chat_history if st.session_state.chat_history else None
        )
        
        st.session_state.current_sentiment = sentiment
        
        if st.session_state.vectorstore:
            if st.session_state.conversation is None:
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
        
        if st.session_state.conversation:
            try:
                response = st.session_state.conversation({"question": question})
                
                if 'chat_history' in response:
                    st.session_state.chat_history = response['chat_history']
                
                if 'source_documents' in response:
                    st.session_state.current_sources = response['source_documents']
                    
                st.session_state.qa_pairs.append({
                    "question": question,
                    "answer": response['answer'],
                    "chunks": st.session_state.current_chunks if hasattr(st.session_state, 'current_chunks') else [],
                    "context": st.session_state.current_context if hasattr(st.session_state, 'current_context') else "",
                    "sentiment": st.session_state.current_sentiment  
                })
            except ValueError as e:
                st.error(f"Error processing question: {str(e)}")
                st.session_state.memory = None
                st.session_state.conversation = None
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
                response = st.session_state.conversation({"question": question})
                
                if 'chat_history' in response:
                    st.session_state.chat_history = response['chat_history']
                
                st.session_state.qa_pairs.append({
                    "question": question,
                    "answer": response['answer'],
                    "chunks": st.session_state.current_chunks if hasattr(st.session_state, 'current_chunks') else [],
                    "context": st.session_state.current_context if hasattr(st.session_state, 'current_context') else "",
                    "sentiment": st.session_state.current_sentiment  
                })
        else:
            st.error("Conversation chain is not initialized. Please check your configuration.")

def display_qa_history():
    """Display all Q&A pairs in the session history with sentiment indicators"""
    for i, qa in enumerate(st.session_state.qa_pairs):
        with st.container():
            sentiment = qa.get("sentiment", "neutral")
            sentiment_icon = {
                "confused": "❓",
                "frustrated": "😤",
                "curious": "🤔",
                "appreciative": "👍",
                "neutral": "💬"
            }.get(sentiment, "💬")
            
            st.markdown(f"### Question {i+1} {sentiment_icon}")
            st.markdown(f"**Q:** {qa['question']}")
            st.markdown(f"**A:** {qa['answer']}")
            
            if 'chunks' in qa and qa['chunks'] and st.session_state.show_chunks:
                with st.expander("Show Retrieved Chunks"):
                    for j, chunk in enumerate(qa['chunks']):
                        st.markdown(f"**Chunk {j+1}:**")
                        st.markdown(f"```\n{chunk.page_content}\n```")
            
            st.markdown("---")

def main():
    st.set_page_config(page_title="CiviBot - Your Civil Engineering Assistant", page_icon=":construction_worker:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "memory" not in st.session_state:
        st.session_state.memory = None
    if "current_sentiment" not in st.session_state:
        st.session_state.current_sentiment = "neutral"
    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "current_chunks" not in st.session_state:
        st.session_state.current_chunks = []
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []
    if "current_context" not in st.session_state:
        st.session_state.current_context = ""
    if "show_chunks" not in st.session_state:
        st.session_state.show_chunks = True
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    st.title("👷‍♂️ Meet CiviBot - Your Civil Engineering Buddy")

    st.markdown("""
    Hello! I'm **CiviBot**, your friendly Civil Engineering assistant powered by Llama 3 70B.  
    I can help explain lectures, clear up concepts, or chat about cement, concrete, and construction.  
    Ask me anything you're stuck on!
    """)

    with st.sidebar:
        st.header("Admin Panel")
        if st.button("Clear Conversation History"):
            st.session_state.qa_pairs = []
            st.session_state.chat_history = None
            st.session_state.memory = None
            st.session_state.conversation = None  
            st.session_state.user_question = ""
            st.rerun()
        
        st.session_state.show_chunks = st.checkbox("Show Retrieved Chunks", value=True)
        
        st.header("Vector Database Info")
        st.write(f"Collection: {COLLECTION_NAME}")
        st.write(f"Embedding Model: {EMBEDDING_MODEL}")
        st.write(f"Chunks per query: {K}")
        
        st.header("Current Session Info")
        if st.session_state.qa_pairs:
            last_sentiment = st.session_state.qa_pairs[-1].get("sentiment", "neutral")
            st.write(f"Last detected sentiment: **{last_sentiment}**")
            
            sentiment_counts = {}
            for qa in st.session_state.qa_pairs:
                sent = qa.get("sentiment", "neutral")
                sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
            
            st.write("Sentiment distribution:")
            for sent, count in sentiment_counts.items():
                st.write(f"- {sent}: {count}")
    
    if st.session_state.conversation is None:
        with st.spinner("Loading vector database and getting things ready..."):
            vectorstore = get_vectorstore()
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("CiviBot is ready to chat!")
            else:
                st.error("Failed to load vector database. Please check your configuration.")

    if st.session_state.qa_pairs:
        st.markdown("## Previous Questions & Answers")
        display_qa_history()

    def submit_question():
        if st.session_state.user_question:
            question = st.session_state.user_question
            st.session_state.user_question = ""
            handle_userinput(question)
            st.rerun()

    user_question = st.text_input(
        "Ask me something from your lecture:", 
        key="user_question",
        on_change=submit_question
    )
    
    if not st.session_state.qa_pairs:
        st.markdown("### 🚀 Try asking me about:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Concrete Properties"):
                handle_userinput("What are the key properties of concrete?")
                st.rerun()
        
        with col2:
            if st.button("Beam Analysis"):
                handle_userinput("How do you analyze a simply supported beam?")
                st.rerun()
                
        with col3:
            if st.button("Foundation Types"):
                handle_userinput("What are the different types of foundations and when to use them?")
                st.rerun()

if __name__ == '__main__':
    main()