import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "distilgpt2"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Predict next word
def predict_next_word(text, top_k=5):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs.logits[:, -1, :]
    predictions = torch.topk(logits, top_k, dim=-1)
    
    predicted_tokens = predictions.indices[0].tolist()
    predicted_words = [tokenizer.decode([token]).strip() for token in predicted_tokens]
    prediction_scores = predictions.values[0].tolist()
    
    return dict(zip(predicted_words, prediction_scores))

# Streamlit app
def main():
    st.title("Next Word Prediction")
    
    text = st.text_area("Enter some text:", value="Hello how")
    top_k = st.number_input("Number of predictions:", min_value=1, max_value=10, value=5)
    
    if st.button("Predict Next Word"):
        preds = predict_next_word(text, top_k)
        
        st.subheader("Predicted Next Words:")
        for word, score in preds.items():
            st.write(f"{word}: {score:.3f}")

if __name__ == "__main__":
    main()
