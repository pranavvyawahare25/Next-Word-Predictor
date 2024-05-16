import streamlit as st
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)

# Set up the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Define a function to predict the next word
@st.cache_data
def predict_next_word(text, top_k=5):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    output = model(input_ids)[0][:, -1, :]
    predictions = torch.topk(output, top_k, dim=-1)
    predicted_tokens = predictions.indices.tolist()[0]
    predicted_words = [tokenizer.decode(token).strip() for token in predicted_tokens]
    prediction_scores = predictions.values.tolist()[0]
    
    next_word_predictions = dict(zip(predicted_words, prediction_scores))
    return next_word_predictions

# Streamlit app
def main():
    st.title("Next Word Prediction")
    
    # Get user input
    text = st.text_area("Enter some text:", value="Hello how")
    top_k = st.number_input("Number of predictions:", min_value=1, max_value=10, value=5, step=1)
    
    if st.button("Predict Next Word"):
        # Make predictions
        next_word_predictions = predict_next_word(text, top_k)
        
        # Display predictions
        st.subheader("Predicted Next Words:")
        for word, score in next_word_predictions.items():
            st.write(f"{word}: {score:.3f}")

if __name__ == "__main__":
    main()
