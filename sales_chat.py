import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv
import os
import openai
from typing import List, Tuple, Dict


def load_dataset(csv_path: str = 'RTC.csv') -> Tuple[pd.DataFrame, LabelEncoder]:
    """Load and preprocess RTC.csv dataset."""
    try:
        dataset = pd.read_csv(csv_path, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    except FileNotFoundError:
        print(f"❌ Error: Could not find RTC.csv at {csv_path}")
        raise

    if dataset.isnull().values.any():
        print("⚠️ Missing values detected. Dropping rows.")
        dataset = dataset.dropna()

    print(f"📦 Dataset loaded: {len(dataset)} samples")
    print("Intent distribution:")
    print(dataset['Intent'].value_counts())

    le = LabelEncoder()
    dataset['label'] = le.fit_transform(dataset['Intent'])
    print(f"🔢 Intents: {le.classes_}")
    return dataset, le


def get_embedding(text: str, embed_model: SentenceTransformer) -> np.ndarray:
    """Generate semantic embedding for a given text."""
    embeddings = embed_model.encode([text], convert_to_tensor=False)
    return embeddings[0]


def get_gpt_response(trigger: str, chat_history: List[dict], predicted_intent: str, conversation_state: Dict) -> str:
    """Generate a response using GPT when similarity is low."""
    try:
        # Set OpenAI API key
        # openai.api_key = os.getenv('OPENAI_API_KEY')  # Replace with your key
        openai.api_key = os.environ["OPENAI_API_KEY"]

        # System prompt to guide GPT
        # system_prompt = (
        #     "You are a friendly and persuasive sales agent for Senior Benefits, a division of US Life, "
        #     "offering final expense insurance. Respond to the user's input in a professional, empathetic, "
        #     "and engaging tone, aligning with the predicted intent (Rebuttal, Concern, or Transfer Line) "
        #     f"and conversation stage ({conversation_state['stage']}). Use the chat history to maintain context. "
        #     "For Rebuttal, address objections and offer a no-cost quote. For Concern, provide clear information "
        #     "about the company or service. For Transfer Line, prepare to connect to a senior agent."
        # )
        #
        # # Construct messages with history and current trigger
        # messages = [
        #     {"role": "system", "content": system_prompt},
        #     *chat_history[-10:],  # Limit to last 10 exchanges
        #     {"role": "user",
        #      "content": f"Intent: {predicted_intent}\nStage: {conversation_state['stage']}\nUser: {trigger}"}
        # ]

        formatted_history = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-10:]
        ])

        # Step 2: Embed everything into the system prompt
        system_prompt = (
            "You are a friendly and persuasive sales agent for Senior Benefits, a division of US Life, "
            "offering final expense insurance. Respond to the user's input in a professional, empathetic, "
            "and engaging tone, aligning with the predicted intent (Rebuttal, Concern, or Transfer Line) "
            f"and conversation stage ({conversation_state['stage']}).\n\n"
            "Use the following chat history to maintain context:\n\n"
            f"{formatted_history}\n\n"
            f"Current Intent: {predicted_intent}\n"
            f"Current Stage: {conversation_state['stage']}\n"
            f"User Message: {trigger}\n\n"
            "For Rebuttal, address objections and offer a no-cost quote. "
            "For Concern, provide clear information about the company or service. "
            "For Transfer Line, prepare to connect to a senior agent.\n\n"
            "**Keep your reply concise—1 to 3 sentences maximum.**"
        )

        # print(system_prompt)
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ Error calling OpenAI API: {e}")
        return "Sorry, I couldn't generate a response. Please try again."


def append_to_rtc_csv(trigger: str, response: str, intent: str, filename: str = 'RTC.csv'):
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write headers only if the file is new
        if not file_exists:
            writer.writerow(['Trigger', 'Response', 'Intent'])

        writer.writerow([trigger, response, intent])


def predict_intent_and_response(trigger: str, model: DistilBertForSequenceClassification,
                                tokenizer: DistilBertTokenizer, le: LabelEncoder,
                                dataset: pd.DataFrame, embed_model: SentenceTransformer,
                                chat_history: List[dict], conversation_state: Dict) -> Tuple[str, str]:
    """Predict intent and select the best response using the trained model or GPT."""
    # Normalize trigger for robust rule-based matching
    trigger_lower = trigger.lower().strip()

    # Rule-based overrides for specific triggers
    if trigger_lower == "what do you sell":
        response = "I'm calling from Senior Benefits, a division of US Life, which offers final expense insurance to cover funeral costs and more."
        return "Concern", response
    if trigger_lower == "how are you":
        response = "I'm calling from Senior Benefits, a division of US Life. We're here to help with your insurance needs."
        return "Concern", response

    # Predict intent using DistilBERT
    inputs = tokenizer(trigger, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    predicted_intent = le.inverse_transform([predicted_label])[0]

    # Filter dataset for predicted intent
    intent_data = dataset[dataset['Intent'] == predicted_intent]
    if intent_data.empty:
        return predicted_intent, "Sorry, I don't understand."

    # Select response using cosine similarity
    trigger_embedding = get_embedding(trigger, embed_model)
    similarities = []
    for dataset_trigger in intent_data['Trigger']:
        dataset_embedding = get_embedding(dataset_trigger, embed_model)
        similarity = cosine_similarity([trigger_embedding], [dataset_embedding])[0][0]
        similarities.append(similarity)

    # Pick response with highest similarity, or use GPT if below threshold
    best_idx = np.argmax(similarities)
    if similarities[best_idx] < 0.7:
        print(f"⚠️ Low similarity ({similarities[best_idx]:.4f}). Using GPT for response.")
        response = get_gpt_response(trigger, chat_history, predicted_intent, conversation_state)
        append_to_rtc_csv(trigger, response, predicted_intent)
        return predicted_intent, response
    response = intent_data.iloc[best_idx]['Response']
    return predicted_intent, response


def main():
    """Run an interactive chatbot using the trained sales model with chat history."""
    # Paths and constants
    AGENT_NAME = "Bill"  # Change to desired agent name
    model_path = 'my_sales_model'
    csv_path = r'C:\Users\HORIZON\launcher\DistilbirtChat\RTC.csv'

    # Initialize chat history and conversation state
    chat_history = []
    conversation_state = {"stage": "greeting"}

    # Load dataset
    try:
        dataset, le = load_dataset(csv_path)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Load model and tokenizer
    try:
        print(f"✅ Loading model and tokenizer from {model_path}")
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(
            f"❌ Error: Could not load model or tokenizer from {model_path}. Please ensure the model is trained and saved. Error: {e}")
        return

    # Load sentence transformer
    try:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
        print("✅ SentenceTransformer loaded")
    except Exception as e:
        print(f"❌ Error loading SentenceTransformer: {e}")
        return

    # Start chat with greeting
    greeting = f"Hi, this is {AGENT_NAME} from Senior Benefits. How are you doing today?"
    print(f"🤖 {greeting}\n")
    chat_history.append({"role": "assistant", "content": greeting})

    # Get first user input
    user_input = input("You: ").strip()
    if user_input.lower() in ['exit', 'quit']:
        print("🤖 Goodbye! Thank you for chatting with Senior Benefits.")
        return
    if not user_input:
        print("🤖 Please enter a valid message.")
        return

    # Append first user input to history
    chat_history.append({"role": "user", "content": user_input})

    # Deliver pitch and update state
    pitch = (
        "I am calling to inform you about a Brand-New State Approved Final Expense Whole Life Insurance Plan.\n"
        "This Life Insurance Plan is designed to cover the 100% cost of Funeral Burial or Cremation Expenses and these Plans are designed for people on Fixed Income, Social Security or Disability income.\n"
        "So I do have a Licensed agent who would like to provide a quick quote over the phone right now I just wanted to make sure how old are you today?"
    )
    print(f"🤖 {pitch}\n")
    chat_history.append({"role": "assistant", "content": pitch})
    conversation_state["stage"] = "pitch"

    # Interactive chat loop


    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("🤖 Goodbye! Thank you for chatting with Senior Benefits.")
            break
        if user_input.lower() == 'history':
            print("\n📜 Chat History:")
            for entry in chat_history:
                print(f"{entry['role'].capitalize()}: {entry['content']}")
            print()
            continue
        if not user_input:
            print("🤖 Please enter a valid message.")
            continue

        # Predict intent and response
        try:
            intent, response = predict_intent_and_response(
                user_input, model, tokenizer, le, dataset, embed_model, chat_history, conversation_state
            )
            print(f"🤖 Intent: {intent}")
            print(f"🤖 Response: {response}\n")

            # Append to chat history
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"❌ Error processing input: {e}")
            print("🤖 Please try again.\n")


if __name__ == "__main__":
    main()