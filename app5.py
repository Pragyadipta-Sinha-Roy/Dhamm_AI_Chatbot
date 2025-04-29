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

BLOOMS_TAXONOMY = {
    "remember": {
        "description": "Recall facts and basic concepts",
        "verbs": ["define", "list", "name", "identify", "recall", "state", "what", "who", "when", "where"]
    },
    "understand": {
        "description": "Explain ideas or concepts",
        "verbs": ["explain", "describe", "interpret", "summarize", "discuss", "clarify", "how", "why"]
    },
    "apply": {
        "description": "Use information in new situations",
        "verbs": ["apply", "demonstrate", "calculate", "solve", "use", "illustrate", "show"]
    },
    "analyze": {
        "description": "Draw connections among ideas",
        "verbs": ["analyze", "compare", "contrast", "distinguish", "examine", "differentiate", "relationship"]
    },
    "evaluate": {
        "description": "Justify a stand or decision",
        "verbs": ["evaluate", "assess", "critique", "judge", "defend", "argue", "support", "recommend", "best"]
    },
    "create": {
        "description": "Produce new or original work",
        "verbs": ["create", "design", "develop", "propose", "construct", "formulate", "devise", "invent"]
    }
}

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

def detect_cognitive_level(text):
    text_lower = text.lower()
    
    for level, info in reversed(BLOOMS_TAXONOMY.items()):
        if any(verb in text_lower.split() for verb in info["verbs"]):
            return level
    
    return "understand"

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

def generate_bloom_specific_prompt(cognitive_level, sentiment):
    base_template = """
    You are CiviBot, a helpful and knowledgeable assistant specializing in civil engineering concepts. Your primary goal is to help students understand their lecture material by providing clear, accurate explanations about civil engineering topics.

    ## Your Knowledge Base
    - You have access to a repository of civil engineering lecture transcripts.
    - You can retrieve relevant information from these transcripts to answer questions.
    - If asked about something outside your knowledge base, acknowledge the limitations and offer to help with what you do know.

    ## User's Cognitive Level and Learning Needs
    The user's question has been analyzed and identified as belonging to the "{cognitive_level}" level of Bloom's Taxonomy.
    
    This means the user is asking for help with: {cognitive_description}
    
    Based on this cognitive level:
    """
    
    bloom_specific_instructions = {
        "remember": """
    - Focus on providing clear, factual information from the lecture notes
    - Define key terms precisely and concisely
    - List relevant information in an organized manner
    - Provide direct answers to factual questions
    - Include specific examples from lecture materials when relevant
    """,
        "understand": """
    - Explain concepts in your own words, avoiding technical jargon when possible
    - Provide analogies or real-world examples to illustrate concepts
    - Compare and contrast related ideas to enhance understanding
    - Rephrase complex ideas in simpler terms
    - Summarize key points from the lecture materials
    """,
        "apply": """
    - Demonstrate how concepts can be applied to solve problems
    - Provide step-by-step procedures for calculations or processes
    - Use real-world civil engineering scenarios to illustrate applications
    - Include worked examples that show how to apply formulas or principles
    - Suggest practice problems that reinforce application skills
    """,
        "analyze": """
    - Break down complex concepts into their constituent parts
    - Highlight relationships between different engineering principles
    - Compare and contrast different methodologies or approaches
    - Discuss cause-effect relationships in civil engineering contexts
    - Help the student see patterns or organizational principles in the material
    """,
        "evaluate": """
    - Present multiple perspectives or approaches to civil engineering problems
    - Discuss pros and cons of different methodologies
    - Help the student develop criteria for making engineering judgments
    - Encourage critical thinking about standard practices
    - Assess the validity of different claims or methods in context
    """,
        "create": """
    - Support innovative thinking and problem-solving
    - Provide frameworks for designing new solutions
    - Discuss how existing principles might be combined in novel ways
    - Encourage theoretical exploration of new ideas
    - Guide the student's creative process without imposing limits
    """
    }
    
    sentiment_instructions = {
        "neutral": """
    ## User Sentiment
    The user appears to be in a neutral state.
    - Maintain a professional, informative tone
    - Focus on delivering accurate content at the appropriate cognitive level
    """,
        "confused": """
    ## User Sentiment
    The user appears to be confused or uncertain.
    - Use simpler language and avoid complex terminology
    - Break down concepts into smaller, more manageable parts
    - Provide more examples to illustrate points
    - Check for understanding by summarizing key points
    - Offer alternative explanations for difficult concepts
    """,
        "frustrated": """
    ## User Sentiment
    The user appears to be frustrated.
    - Acknowledge their difficulty and provide reassurance
    - Offer multiple approaches to understanding the concept
    - Use very clear, step-by-step explanations
    - Emphasize that many students find this challenging
    - Focus on building confidence alongside understanding
    """,
        "curious": """
    ## User Sentiment
    The user appears to be curious and engaged.
    - Match their enthusiasm in your response
    - Provide additional interesting details beyond the basics
    - Suggest related topics they might find interesting
    - Connect the current topic to broader civil engineering concepts
    - Encourage further exploration with additional questions
    """
    }
    
    closing_template = """
    ## Response Guidelines
    - Keep explanations concise but complete
    - Use bullet points for lists of steps or related concepts
    - Format mathematical equations clearly when needed
    - Refer to specific sections of lectures when relevant
    - IMPORTANT: Always refer to previous conversation context when appropriate
    - Always maintain continuity with previous answers
    - Always end with an offer to help further or to support progression to the next cognitive level
    
    Remember: Your goal is to help students understand civil engineering concepts at their current cognitive level, while encouraging growth to higher levels of thinking.

    ## Relevant Context from lecture transcripts:
    {context}
    
    ## Current Question: 
    {question}
            
    Helpful Response:
    """
    
    full_template = (
        base_template.format(
            cognitive_level=cognitive_level,
            cognitive_description=BLOOMS_TAXONOMY[cognitive_level]["description"]
        ) + 
        bloom_specific_instructions[cognitive_level] +
        sentiment_instructions[sentiment] +
        closing_template
    )
    
    return PromptTemplate(
        template=full_template,
        input_variables=["context", "question"]
    )

def get_conversation_chain(vectorstore, cognitive_level="understand", sentiment="neutral"):
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
    
    prompt = generate_bloom_specific_prompt(cognitive_level, sentiment)
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": K}),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=True
    )

def lookup_relevant_chunks(query, vectorstore):
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
        
        cognitive_level = detect_cognitive_level(question)
        
        st.session_state.current_sentiment = sentiment
        st.session_state.current_cognitive_level = cognitive_level
        
        if st.session_state.vectorstore:
            if st.session_state.conversation is None:
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
        
        if st.session_state.conversation:
            try:
                bloom_prompt = generate_bloom_specific_prompt(cognitive_level, sentiment)
                
                st.session_state.conversation.combine_docs_chain.llm_chain.prompt = bloom_prompt
                
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
                    "cognitive_level": cognitive_level,
                    "sentiment": sentiment
                })
            except ValueError as e:
                st.error(f"Error processing question: {str(e)}")
                st.session_state.memory = None
                st.session_state.conversation = None
                st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
                
                bloom_prompt = generate_bloom_specific_prompt(cognitive_level, sentiment)
                st.session_state.conversation.combine_docs_chain.llm_chain.prompt = bloom_prompt
                
                response = st.session_state.conversation({"question": question})
                
                if 'chat_history' in response:
                    st.session_state.chat_history = response['chat_history']
                
                st.session_state.qa_pairs.append({
                    "question": question,
                    "answer": response['answer'],
                    "chunks": st.session_state.current_chunks if hasattr(st.session_state, 'current_chunks') else [],
                    "context": st.session_state.current_context if hasattr(st.session_state, 'current_context') else "",
                    "cognitive_level": cognitive_level,
                    "sentiment": sentiment
                })
        else:
            st.error("Conversation chain is not initialized. Please check your configuration.")

def display_qa_history():
    for i, qa in enumerate(st.session_state.qa_pairs):
        with st.container():
            st.markdown(f"### Question {i+1}")
            st.markdown(f"**Q:** {qa['question']}")
            
            if 'cognitive_level' in qa:
                level = qa['cognitive_level']
                st.markdown(f"*Cognitive Level: {level.capitalize()} - {BLOOMS_TAXONOMY[level]['description']}*")
            
            if 'sentiment' in qa:
                st.markdown(f"*Sentiment: {qa['sentiment'].capitalize()}*")
                
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
    if "current_cognitive_level" not in st.session_state:
        st.session_state.current_cognitive_level = "understand"
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
    if "show_cognitive_levels" not in st.session_state:
        st.session_state.show_cognitive_levels = True
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
        st.session_state.show_cognitive_levels = st.checkbox("Show Bloom's Taxonomy Levels", value=True)
        
        st.header("Vector Database Info")
        st.write(f"Collection: {COLLECTION_NAME}")
        st.write(f"Embedding Model: {EMBEDDING_MODEL}")
        st.write(f"Chunks per query: {K}")
        
        st.header("Bloom's Taxonomy Guide")
        for level, info in BLOOMS_TAXONOMY.items():
            with st.expander(f"{level.capitalize()} - {info['description']}"):
                st.write("Example keywords: " + ", ".join(info["verbs"][:5]) + "...")
    
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

if __name__ == '__main__':
    main()