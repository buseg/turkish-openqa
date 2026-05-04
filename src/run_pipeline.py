import faiss
from datasets import load_from_disk
from retriever import TurkishDPR
from reader import FiDReader

class OpenQAPipeline:
    def __init__(self, data_path, index_path):
        print("Initializing OpenQA Pipeline...")
        
        # 1. Load the original text data to fetch the actual passage strings
        print("Loading knowledge source...")
        self.knowledge_source = load_from_disk(data_path)
        self.passages = self.knowledge_source['text']
        self.titles = self.knowledge_source['title']
        
        # 2. Initialize Retriever and Reader
        self.retriever = TurkishDPR()
        self.reader = FiDReader()
        
        # 3. Load the pre-computed FAISS index into the retriever
        print("Loading FAISS index...")
        self.retriever.index = faiss.read_index(index_path)
        print("Pipeline ready!\n" + "="*50)

    def ask(self, question, top_k=5):
        print(f"Question: {question}")
        
        # Step 1: Retrieve
        print(f"Retrieving top {top_k} passages...")
        distances, indices = self.retriever.retrieve(question, k=top_k)
        
        # Fetch the actual text for the retrieved indices
        retrieved_passages = []
        for idx in indices:
            # Combine title and text for the reader's context
            title = self.titles[idx]
            text = self.passages[idx]
            retrieved_passages.append(f"{title}. {text}")
            
        # Step 2: Read/Generate
        print("Synthesizing answer with FiD...")
        answer = self.reader.generate_answer(question, retrieved_passages)
        
        print("\n" + "="*50)
        print(f"FINAL ANSWER: {answer}")
        print("="*50 + "\n")
        
        return answer, retrieved_passages

# --- EXECUTION ---
if __name__ == "__main__":
    DATA_PATH = "./cluster_data/final_knowledge_source_chunked"
    INDEX_PATH = "./cluster_data/wikipedia_tr_dpr.index"
    
    # Initialize the full pipeline
    qa_system = OpenQAPipeline(DATA_PATH, INDEX_PATH)
    
    # Let's test it end-to-end!
    test_question = "Amazon Havzası'nda kaç ülke bulunmaktadır?"
    
    # We use top_k=5 as specified for the knowledge selector extension later in your proposal
    qa_system.ask(test_question, top_k=5)