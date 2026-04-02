import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

# Model name
model_name = "distilgpt2"

# Load model (cached)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    return tokenizer, model, device

tokenizer, model, device = load_model()

# Prediction function
def predict_next_word(text, top_k=5):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    logits = outputs.logits[:, -1, :]
    
    # Convert logits → probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get top k predictions
    top_probs, top_indices = torch.topk(probs, top_k)
    
    predicted_tokens = top_indices[0].tolist()
    predicted_words = [
        tokenizer.decode([token]).strip() for token in predicted_tokens
    ]
    prediction_scores = top_probs[0].tolist()
    
    return list(zip(predicted_words, prediction_scores))


# Streamlit UI
def main():
    st.set_page_config(page_title="Next Word Prediction", layout="centered")
    
    st.title("🔮 Next Word Prediction")
    
    # Input
    text = st.text_area("Enter some text:", value="Hello how")
    top_k = st.number_input(
        "Number of predictions:", min_value=1, max_value=10, value=5
    )
    
    if st.button("Predict Next Word"):
        if text.strip() == "":
            st.warning("Please enter some text.")
            return
        
        predictions = predict_next_word(text, top_k)
        
        st.subheader("📊 Predicted Next Words:")
        
        for word, score in predictions:
            st.write(f"👉 **{word}** → {score*100:.2f}%")

# Run app
if __name__ == "__main__":
    main()
