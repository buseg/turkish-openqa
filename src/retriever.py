import torch
import faiss
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class TurkishDPR:
    def __init__(self, model_name="dbmdz/convbert-base-turkish-cased", device=None):
        # Automatically detect GPU if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing TurkishDPR on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize separate encoders as per DPR architecture
        self.passage_encoder = AutoModel.from_pretrained(model_name).to(self.device)
        self.question_encoder = AutoModel.from_pretrained(model_name).to(self.device)
        
        # FAISS index (Inner Product for DPR)
        self.vector_dim = self.passage_encoder.config.hidden_size 
        self.index = faiss.IndexFlatIP(self.vector_dim)

    def encode_passages(self, titles, texts, batch_size=32):
        """Encodes passages along with their titles for the FAISS index."""
        self.passage_encoder.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding passages"):
                batch_titles = titles[i:i+batch_size]
                batch_texts = texts[i:i+batch_size]
                
                # Passing both lists creates the proper [CLS] Title [SEP] Text [SEP] sequence
                inputs = self.tokenizer(
                    batch_titles, 
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=150, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Use the [CLS] token representation as the dense vector
                outputs = self.passage_encoder(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
                
        return np.vstack(all_embeddings)

    def retrieve(self, query, k=3):
        """Encodes a single query and searches the internal FAISS index."""
        self.question_encoder.eval()
        
        inputs = self.tokenizer(
            query, 
            padding=True, 
            truncation=True, 
            max_length=150, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.question_encoder(**inputs)
            # Extract the [CLS] token representation
            q_embed = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Normalize the query vector to match L2-normalized passage vectors
        faiss.normalize_L2(q_embed)

        # Search the index
        distances, indices = self.index.search(q_embed, k)
        return distances[0], indices[0]


def verify_retrieval(dpr_model, titles, passages, query="Türkiye'nin başkenti neresidir?", k=3):
    """Utility function to test the index immediately after building."""
    print(f"\n--- Running Verification ---")
    print(f"Query: '{query}'")
    print(f"Searching for top {k} results...\n")
    
    distances, indices = dpr_model.retrieve(query, k=k)

    for rank, (dist, idx) in enumerate(zip(distances, indices)):
        print(f"Rank {rank + 1} | Distance Score: {dist:.4f} | Index: {idx}")
        print(f"Title: {titles[idx]}")
        print(f"Passage: {passages[idx]}")
        print("-" * 50)


# --- EXECUTION ---
if __name__ == "__main__":
    print("Loading chunked knowledge source...")
    knowledge_source = load_from_disk("odqa_data/final_knowledge_source_chunked")
    
    # Extract both lists
    titles = knowledge_source['title']
    passages = knowledge_source['text']
    
    # For testing, you can slice the data here (e.g., titles[:1000], passages[:1000])
    # titles = titles[:1000]
    # passages = passages[:1000]
    
    dpr = TurkishDPR()
    
    print("Computing passage embeddings offline...")
    passage_embeddings = dpr.encode_passages(titles, passages, batch_size=256)
    
    print("Adding embeddings to FAISS index...")
    faiss.normalize_L2(passage_embeddings)
    dpr.index.add(passage_embeddings)
    print(f"Index successfully built with {dpr.index.ntotal} vectors.")
    
    print("Saving FAISS index to disk...")
    faiss.write_index(dpr.index, "odqa_data/wikipedia_tr_dpr.index")
    print("Indexing complete.")
    
    # Run the sanity check using the index in memory
    verify_retrieval(dpr, titles, passages)