import nltk
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random

try:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("punkt_tab")
except:
    print("NLTK data already downloaded")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    message: str


class MedicineChatbot:
    def __init__(self):
        self.QA = [
            [
                r"what is (aspirin|paracetamol|ibuprofen|amoxicillin|metformin|lisinopril|atorvastatin)\??",
                [
                    "%1 is a medicine used to treat pain, fever, and inflammation.",
                    "Yes, %1 is commonly used for various conditions, including pain relief and reducing fever.",
                ],
            ],
            [
                r"side effects of (aspirin|paracetamol|ibuprofen|amoxicillin|metformin|lisinopril|atorvastatin)\??",
                [
                    "Common side effects of %1 include nausea, vomiting, and stomach pain. Always consult your doctor for personalized advice.",
                    "Side effects of %1 may include dizziness, headache, and allergic reactions in some patients.",
                ],
            ],
            [
                r"how to take (aspirin|paracetamol|ibuprofen|amoxicillin|metformin|lisinopril|atorvastatin)\??",
                [
                    "You can take %1 as directed by your physician, usually with food to minimize stomach upset.",
                    "It's recommended to take %1 with a full glass of water, unless your doctor instructs otherwise.",
                ],
            ],
            [
                r"dosage of (aspirin|paracetamol|ibuprofen|amoxicillin|metformin|lisinopril|atorvastatin)\??",
                [
                    "The dosage of %1 depends on your age, weight, and the condition being treated. Please refer to your healthcare provider for the right dosage.",
                    "For adults, the common dosage of %1 varies, but do not exceed the recommended limit set by your doctor.",
                ],
            ],
            [
                r"when to take (aspirin|paracetamol|ibuprofen|amoxicillin|metformin|lisinopril|atorvastatin)\??",
                [
                    "It's best to take %1 at the same time every day for consistent results.",
                    "Take %1 as prescribed, often at regular intervals throughout the day.",
                ],
            ],
            [
                r"what does (aspirin|paracetamol|ibuprofen|amoxicillin|metformin|lisinopril|atorvastatin) do\??",
                [
                    "%1 is primarily used for treating pain relief, reducing inflammation, or managing chronic conditions.",
                    "%1 helps lower blood pressure, control blood sugar, or manage cholesterol levels, depending on the medication.",
                ],
            ],
            [
                r"can you tell me about (aspirin|paracetamol|ibuprofen|amoxicillin|metformin|lisinopril|atorvastatin)\??",
                [
                    "%1 is widely used for various health issues. Would you like to know more about its specific uses or side effects?",
                    "Yes, %1 is a medication commonly used for pain and inflammation. It's important to follow your doctor's recommendations.",
                ],
            ],
            [
                r"who should not take (aspirin|paracetamol|ibuprofen|amoxicillin|metformin|lisinopril|atorvastatin)\??",
                [
                    "Individuals with allergies to %1 or its components should avoid it. Always consult a doctor.",
                    "People with certain medical conditions or on specific medications should not take %1 without medical advice.",
                ],
            ],
            [
                r"interactions of (aspirin|paracetamol|ibuprofen|amoxicillin|metformin|lisinopril|atorvastatin)\??",
                [
                    "%1 can interact with several medications, including blood thinners and other anti-inflammatory drugs.",
                    "It's essential to discuss with your healthcare provider about any medications you are currently taking before starting %1.",
                ],
            ],
            [
                r"hello|hi|hey|greetings",
                ["Hello! How can I assist you with your medicine inquiries today?"],
            ],
            [r"quit|exit|goodbye", ["Bye, take care! Hope to see you soon! :)"]],
            [
                r"(.*)",
                [
                    "I'm not sure I understand that. Could you ask something else about medicines?"
                ],
            ],
        ]

    def get_response(self, user_input):
        """Generate response based on user input using regex patterns"""
        user_input = user_input.lower()  # Normalize input

        for pattern, responses in self.QA:
            match = re.search(pattern, user_input)
            if match:
                # Replace %1 with the matched group
                medicine_name = match.group(1)
                response = random.choice(responses)
                return response.replace("%1", medicine_name)

        return "I'm not sure I understand. Could you rephrase that?"


chatbot = MedicineChatbot()


@app.post("/chat")
async def chat(message: ChatMessage):
    try:
        response = chatbot.get_response(message.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
