import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

class FiDReader:
    def __init__(self, model_name="google/mt5-base", device="cuda"):
        self.device = device
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def generate_answer(self, question, retrieved_passages, max_length=50):
        self.model.eval()
        
        # Format inputs: "question: [Q] context: [C]" for each passage
        formatted_inputs = [f"soru: {question} bağlam: {p}" for p in retrieved_passages]
        
        # Tokenize all passages independently
        inputs = self.tokenizer(
            formatted_inputs, 
            padding=True, 
            truncation=True, 
            max_length=250, 
            return_tensors="pt"
        ).to(self.device)

        batch_size = 1 # Processing one question at a time in this function
        num_passages = len(retrieved_passages)
        seq_len = inputs['input_ids'].shape[1]

        with torch.no_grad():
            # Independent encoding of the input
            encoder_outputs = self.model.encoder(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask']
            )
            
            # Flatten the encoded passages into a single sequence
            # Shape goes from (num_passages, seq_len, hidden_dim) -> (1, num_passages * seq_len, hidden_dim)
            hidden_dim = encoder_outputs.last_hidden_state.shape[-1]
            fused_hidden_states = encoder_outputs.last_hidden_state.view(batch_size, num_passages * seq_len, hidden_dim)
            
            # Create a combined attention mask for the decoder
            fused_attention_mask = inputs['attention_mask'].view(batch_size, num_passages * seq_len)
            
            # Wrap in the Hugging Face output format
            combined_encoder_outputs = BaseModelOutput(last_hidden_state=fused_hidden_states)

            # Decoding the fused representation to generate the answer
            output_ids = self.model.generate(
                encoder_outputs=combined_encoder_outputs,
                attention_mask=fused_attention_mask,
                max_length=max_length,
                num_beams=3, # Beam search for better generation
                early_stopping=True
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

# --- TEST EXECUTION ---
if __name__ == "__main__":
    reader = FiDReader()
    
    # A simple question
    sample_question = "Türkiye'nin en yüksek dağı hangisidir?"
    
    # 1 Correct passage, 2 highly related but incorrect distractors
    sample_passages = [
        "Erciyes Dağı, Kayseri ilinde yer alan ve 3.917 metre yüksekliğiyle İç Anadolu'nun en yüksek dağıdır.",
        "Ağrı Dağı, 5.137 metrelik rakımıyla Türkiye'nin en yüksek dağıdır. Türkiye'nin doğusunda, İran sınırına yakın bir konumda yer alır.",
        "Kafkas Dağları, Karadeniz ile Hazar Denizi arasında yer alan, Avrupa ve Asya'nın sınırını oluşturan büyük bir sıradağ sistemidir."
    ]
    
    print("Generating answer with FiD...")
    answer = reader.generate_answer(sample_question, sample_passages)
    print(f"Question: {sample_question}")
    print(f"Predicted Answer: {answer}")