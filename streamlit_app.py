import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Mental Health Companion", page_icon="💙", layout="wide")

# Load models and tokenizers
@st.cache_resource
def load_models():
    encoder = load_model('encoder_model.h5', compile=False)
    decoder = load_model('decoder_model.h5', compile=False)
    
    with open('encoder_tokenizer.pkl', 'rb') as f:
        enc_tokenizer = pickle.load(f)
    
    with open('decoder_tokenizer.pkl', 'rb') as f:
        dec_tokenizer = pickle.load(f)
    
    with open('config.pkl', 'rb') as f:
        config = pickle.load(f)
    
    return encoder, decoder, enc_tokenizer, dec_tokenizer, config

encoder_model, decoder_model, encoder_tokenizer, decoder_tokenizer, config = load_models()

max_encoder_seq_length1 = config['max_encoder_seq_length']
INFERENCE_MAX_RESPONSE_LENGTH = config.get('max_decoder_seq_length', 50)
target_token_index = decoder_tokenizer.word_index
reverse_target_index = {v: k for k, v in target_token_index.items()}

# Your decode_sequence function (EXACT copy)
def decode_sequence(input_seq, repetition_penalty=1.5):
    encoder_output_seq, h, c = encoder_model.predict(input_seq, verbose=0)
    states_value = [h, c]

    start_token_id = target_token_index.get('<start>', 1)
    target_seq = np.array([[start_token_id]])

    decoded_sentence = ""
    stop_condition = False
    generated_tokens = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value + [encoder_output_seq], verbose=0
        )

        penalized_output_tokens = np.copy(output_tokens[0, -1, :])
        for i, token_id in enumerate(generated_tokens):
            if token_id == 0: continue
            penalized_output_tokens[token_id] /= repetition_penalty

        sampled_token_index = np.argmax(penalized_output_tokens)
        sampled_word = reverse_target_index.get(sampled_token_index, "<OOV>")

        if sampled_word == "<end>":
            stop_condition = True
        elif sampled_word == "<PAD>":
            pass
        elif sampled_word != "<OOV>":
            decoded_sentence += " " + sampled_word
            generated_tokens.append(sampled_token_index)

        if len(generated_tokens) >= INFERENCE_MAX_RESPONSE_LENGTH and not stop_condition:
            stop_condition = True

        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    decoded_sentence = decoded_sentence.replace("<PAD>", "").replace("<OOV>", "").replace("<end>", "").strip()
    return decoded_sentence

# Your generate_response function
def generate_response(user_input):
    tokens = encoder_tokenizer.texts_to_sequences([user_input])
    padded_tokens = pad_sequences(tokens, maxlen=max_encoder_seq_length1, padding='post')
    
    chatbot_response = decode_sequence(padded_tokens)
    chatbot_response = chatbot_response.replace("<start>", "").replace("<end>", "").replace("<OOV>", "").replace("<PAD>", "")
    return chatbot_response.strip()

# Streamlit UI
st.title("💙 Mental Health Companion")
st.markdown("*Seq2Seq with Attention - 76.2% Training Accuracy*")

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your mental health companion. How are you feeling today?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Share what's on your mind..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    with st.spinner("Thinking..."):
        response = generate_response(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with st.chat_message("assistant"):
        st.write(response)

# Sidebar
with st.sidebar:
    st.subheader("🤖 Model Info")
    st.markdown("**Accuracy:** 76.2%  \n**Dataset:** 4,167 conversations")
    
    st.markdown("---")
    st.subheader("🆘 Crisis Resources")
    st.markdown("**India:** AASRA - 91-9820466726  \n**USA:** 988 Lifeline")
    
    st.caption("⚠️ Not a replacement for professional care.")