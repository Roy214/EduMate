import streamlit as st
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load fine-tuned Phi-2 model with optimizations
@st.cache_resource
def load_model():
    model_path = "/Users/abhijitroy/Downloads/Edumate_phi/final_model"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,  # Optimized for Mac M3
        device_map="auto"  # Assigns model to best device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

pipe = load_model()

# Probability Topics
PROBABILITY_CONCEPTS = {
    "Basic Probability": {
        "definition": "Probability is a measure of how likely an event is to occur. It ranges from 0 (impossible) to 1 (certain).",
        "example": "The probability of rolling a 6 on a fair 6-sided die is 1/6 ≈ 0.1667."
    },
    "Independent & Dependent Events": {
        "definition": "Independent events do not affect each other’s outcomes, while dependent events do.",
        "example": "Flipping a coin twice (independent) vs. drawing two cards from a deck without replacement (dependent)."
    },
    "Conditional Probability": {
        "definition": "Conditional probability is the probability of an event occurring given that another event has already occurred.",
        "example": "If a bag has 5 red and 3 blue balls, and you draw a red first, the probability of drawing another red changes."
    },
    "Bayes' Theorem": {
        "definition": "Bayes' Theorem describes how to update probabilities based on new evidence.",
        "example": "If a test for a disease is 90% accurate but the disease is rare (1% of people), the probability of having the disease given a positive test is lower than 90%."
    }
}

# Predefined Probability Questions
QUESTION_TEMPLATES = {
    "easy": [
        "What is the probability of rolling a 3 on a fair 6-sided die?",
        "If you flip a fair coin, what is the probability of getting heads?",
        "A bag contains 4 red and 6 blue balls. What is the probability of drawing a red ball?",
        "A die is rolled. What is the probability of getting an even number?",
        "A class has 10 boys and 15 girls. What is the probability of selecting a boy at random?"
    ],
    "medium": [
        "What is the probability of getting two heads when flipping two fair coins?",
        "A box contains 5 green, 3 yellow, and 2 red balls. What is the probability of drawing a yellow ball?",
        "If you roll two dice, what is the probability that the sum is 7?",
        "A deck has 52 cards. What is the probability of drawing a king?",
        "A couple plans to have 3 children. What is the probability of having exactly two boys?"
    ],
    "hard": [
        "A family has 3 children. What is the probability that all are boys?",
        "A fair die is rolled twice. What is the probability of rolling an even number both times?",
        "What is the probability of drawing two aces from a shuffled deck of 52 cards?",
        "A bag contains 5 white, 4 red, and 3 blue balls. If two balls are drawn, what is the probability that both are red?",
        "In a group of 10 people, what is the probability that at least two share the same birthday?"
    ]
}

# Function to generate a single question
def get_probability_question(difficulty):
    return random.choice(QUESTION_TEMPLATES[difficulty])

# Function to validate user's answer
def validate_answer(question, user_answer):
    validation_prompt = f"Question: {question}\nAnswer: {user_answer}\nIs this correct? Provide a Yes or No response."
    
    response = pipe(
        validation_prompt, 
        max_new_tokens=20,  
        num_return_sequences=1,
        temperature=0.5,  
        top_p=0.9,
        top_k=20
    )
    
    model_response = response[0]['generated_text'].strip().lower()
    
    if "yes" in model_response:
        return True, "✅ Correct Answer!"
    else:
        return False, "❌ Incorrect. Try again!"

# Streamlit UI
st.title("Edumate🎲 Probability Learning & Quiz")
st.write("Learn about probability and test your knowledge with questions.")

# **Tabs for Learning & Quiz**
tab1, tab2 = st.tabs(["📚 Learn Probability", "🎯 Probability Quiz"])

# **TAB 1: Learning Section**
with tab1:
    st.header("📖 Learn Probability Concepts")

    # Dropdown to select topic
    topic = st.selectbox("Select a topic to learn:", list(PROBABILITY_CONCEPTS.keys()))

    # Display explanation
    if topic:
        st.subheader(topic)
        st.write(f"**Definition:** {PROBABILITY_CONCEPTS[topic]['definition']}")
        st.write(f"**Example:** {PROBABILITY_CONCEPTS[topic]['example']}")

# **TAB 2: Probability Quiz**
with tab2:
    st.header("🎯 Probability Quiz")

    # Select difficulty level
    difficulty = st.radio("Select Difficulty Level:", ["easy", "medium", "hard"])

    # **Generate a single question**
    if st.button("Generate a Question"):
        st.session_state.question = get_probability_question(difficulty)
        st.session_state.validated = False  # Reset validation state
        st.session_state.user_answer = ""  # Clear previous answer
        st.rerun()

    # **Display Single Question**
    if "question" in st.session_state:
        st.subheader("📌 Question:")
        st.write(st.session_state.question)

        # **User answer input**
        user_answer = st.text_input("Enter your answer:", value=st.session_state.get("user_answer", ""))

        # **Validate Answer**
        if st.button("Submit Answer"):
            is_correct, message = validate_answer(st.session_state.question, user_answer)
            st.session_state.validated = is_correct  # Mark as validated if correct
            st.session_state.user_answer = user_answer  # Store user input

            if is_correct:
                st.success(message)
            else:
                st.error(message)

        # **Clear Answer Button**
        if st.button("Clear Answer"):
            st.session_state.user_answer = ""  # Reset user input
            st.rerun()  # Refresh UI
