from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

class QAEngine:
    """
    Handles the retrieval and generation of answers using an LLM.
    """
    def __init__(self):
        """Initializes the QA engine with the LLM and a prompt template."""
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not found.")

        # Initialize the Google Gemini Pro model for chat-based generation
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=google_api_key,
            temperature=0.1,  # Low temperature for more factual, less creative answers
            convert_system_message_to_human=True
        )
        
        # Define the prompt template to guide the LLM's response
        self.prompt_template = """
        Use the following context to answer the question clearly and concisely.
        If you cannot find the answer in the context, say "I cannot find this information in the provided document."

        Context: {context}
        Question: {question}
        
        Answer:
        """
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    async def get_answer(self, vector_store, question: str) -> str:
        """
        Retrieves relevant context and generates an answer for a given question.
        """
        if not vector_store:
            return "Error: Vector store is not available."
            
        try:
            # Create a retrieval QA chain. This chain does the following:
            # 1. Takes the user's question.
            # 2. Searches the vector_store for the most relevant document chunks (retrieval).
            # 3. Stuffs these chunks into the prompt's {context}.
            # 4. Sends the filled prompt to the LLM to generate the final answer.
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # "stuff" means all retrieved docs are stuffed into the context
                retriever=vector_store.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 chunks
                chain_type_kwargs={"prompt": self.prompt},
                return_source_documents=False # We only need the final answer
            )
            
            # Run the chain with the user's question
            result = qa_chain({"query": question})
            return result.get("result", "No answer could be generated.")
            
        except Exception as e:
            print(f"Error during question processing: {e}")
            return f"Error processing question: {str(e)}"
