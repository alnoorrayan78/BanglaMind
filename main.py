import uuid
from datetime import datetime
import json
import os
import string
import re

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# Google GenAI SDK ইমপোর্ট (অফিশিয়াল লাইব্রেরি)
from google import genai
from google.genai import types

app = Flask(__name__)

# =====================================
# GEMINI API SETUP
# =====================================
# আপনার জেমিনি ফ্রি এপিআই কি (API Key) এখানে বসান:
GEMINI_API_KEY = "AQ.Ab8RN6JUtH7uDbxjurLqQ63H4BkJeZyvCIqXwk1Ab3QETBSF-w"

# ক্লায়েন্ট ইনিশিয়ালাইজ করা
client = genai.Client(api_key=GEMINI_API_KEY)


# =====================================
# 1. KNOWLEDGE & MEMORY DATABASE SETUP
# =====================================

BRAIN_FOLDER = "brain"
MEMORY_FOLDER = "memory"
UPLOAD_FOLDER = "uploads"

KNOWLEDGE_FILE = os.path.join(BRAIN_FOLDER, "knowledge.json")
HISTORY_FILE = os.path.join(MEMORY_FOLDER, "chats.json")

os.makedirs(BRAIN_FOLDER, exist_ok=True)
os.makedirs(MEMORY_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

if not os.path.exists(KNOWLEDGE_FILE):
    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


def load_knowledge():
    try:
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_knowledge(data):
    temp_file = KNOWLEDGE_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(temp_file, KNOWLEDGE_FILE)


def load_chats():
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_chats(chats):
    temp_file = HISTORY_FILE + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=4)
    os.replace(temp_file, HISTORY_FILE)


def find_chat(chats, chat_id):
    for chat in chats:
        if chat.get("id") == chat_id:
            return chat
    return None


# =====================================
# 2. TEXT NORMALIZATION & NLP UTILS
# =====================================

def normalize_text(text):
    text = text.strip().lower()
    punctuation = string.punctuation + "।"
    text = text.translate(str.maketrans("", "", punctuation))
    return " ".join(text.split())


def detect_language(text):
    for char in text:
        if "\u0980" <= char <= "\u09FF":
            return "bn"
    return "en"


# =====================================
# 3. BASE CLASSIFIER & TRAINING DATA
# =====================================

training_data = [
    ("হ্যালো", "greeting"),
    ("হাই", "greeting"),
    ("কেমন আছো", "greeting"),
    ("Hello", "greeting"),
    ("Hi", "greeting"),

    ("তোমার নাম কি", "identity"),
    ("তুমি কে", "identity"),
    ("What is your name", "identity"),
    ("Who are you", "identity"),

    ("ধন্যবাদ", "thanks"),
    ("Thank you", "thanks"),

    ("সাহায্য করো", "help"),
    ("Can you help me", "help"),

    ("বিদায়", "goodbye"),
    ("Goodbye", "goodbye"),
]

texts = [x[0] for x in training_data]
labels = [x[1] for x in training_data]

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), lowercase=True)
X = vectorizer.fit_transform(texts)
model = LogisticRegression(max_iter=1000)
model.fit(X, labels)

answers = {
    "greeting": {
        "bn": "হ্যালো! আমি BanglaMind AI—আপনার জেমিনি চালিত সহকারী।",
        "en": "Hello! I am BanglaMind AI—your Gemini powered assistant."
    },
    "identity": {
        "bn": "আমার নাম BanglaMind AI। আমি Google Gemini API দ্বারা চালিত।",
        "en": "My name is BanglaMind AI. I am powered by Google Gemini API."
    },
    "thanks": {
        "bn": "তোমাকে স্বাগতম!",
        "en": "You're welcome!"
    },
    "help": {
        "bn": "অবশ্যই! কোড লেখা, ওয়েবসাইট ডিজাইন বা তথ্য শেখানোর জন্য বলুন।",
        "en": "Of course! Ask me for code, web design, or to teach me facts."
    },
    "goodbye": {
        "bn": "আবার দেখা হবে!",
        "en": "See you again!"
    }
}


# =====================================
# 4. AUTOMATIC FACT EXTRACTION & LEARNING
# =====================================

def extract_and_store_fact(user_message, chat_id):
    msg = user_message.strip()
    norm_msg = normalize_text(msg)
    knowledge = load_knowledge()
    now = datetime.now().isoformat()

    correction_keywords = ["আসলে", "ভুল", "not", "actually", "modify", "update", "পরিবর্তন"]
    is_correction = any(k in norm_msg for k in correction_keywords)

    match_my = re.search(r"আমার\s+([\u0980-\u09ff\w\s]+?)\s*(?:হলো|হল|হিসাবে|হিসেবে)?\s*([^\.?!]+)", msg)
    
    subj, relation, value = None, "is", None

    if match_my:
        parts = msg.replace("আমার", "").strip().split()
        if len(parts) >= 2:
            subj = "user_" + parts[0]
            value = " ".join(parts[1:]).replace("হলো", "").replace("হল", "").strip()
            relation = "preference_or_attribute"
    
    if not value and " রাজধানী " in msg:
        parts = msg.split(" রাজধানী ")
        if len(parts) == 2:
            subj = parts[0].strip() + "_capital"
            relation = "capital"
            value = parts[1].replace("হলো", "").replace("হল", "").strip()

    if not value and not msg.endswith("?") and not msg.endswith(" কি") and not msg.endswith(" কী"):
        if " হলো " in msg or " হল " in msg:
            splitter = " হলো " if " হলো " in msg else " হল "
            parts = msg.split(splitter)
            if len(parts) == 2:
                subj = parts[0].strip()
                relation = "is"
                value = parts[1].strip()

    if subj and value:
        value = value.rstrip(".,?!")
        existing_item = None
        for item in knowledge:
            if item.get("subj") == subj or normalize_text(item.get("question", "")) == normalize_text(subj):
                existing_item = item
                break

        if existing_item:
            old_val = existing_item.get("value") or existing_item.get("answer")
            if old_val != value:
                if "history" not in existing_item:
                    existing_item["history"] = []
                existing_item["history"].append({
                    "old_value": old_val,
                    "updated_at": now,
                    "source_chat_id": chat_id
                })
            existing_item["value"] = value
            existing_item["answer"] = value
            existing_item["original_sentence"] = msg
            existing_item["updated_at"] = now
            save_knowledge(knowledge)
            
            if is_correction or (old_val != value):
                return f"ঠিক আছে, তোমার {subj.replace('user_', '')} হিসেবে '{value}' আপডেট করলাম।"
            return f"বুঝেছি, {value}।"
        else:
            new_entry = {
                "id": str(uuid.uuid4()),
                "subj": subj,
                "relation": relation,
                "value": value,
                "question": subj.replace("user_", "আমার "),
                "answer": value,
                "original_sentence": msg,
                "created_date": now,
                "confidence_score": 0.90,
                "source_conversation_id": chat_id,
                "history": []
            }
            knowledge.append(new_entry)
            save_knowledge(knowledge)
            return f"বুঝেছি, {value}। মনে রাখলাম।"

    return None


def search_knowledge_intelligent(message):
    knowledge = load_knowledge()
    if not knowledge:
        return None

    norm_msg = normalize_text(message)

    for item in knowledge:
        q_norm = normalize_text(item.get("question", ""))
        subj_norm = normalize_text(item.get("subj", ""))
        if q_norm == norm_msg or subj_norm == norm_msg or norm_msg in q_norm:
            return item.get("answer") or item.get("value")

    if "নাম" in norm_msg or "name" in norm_msg:
        for item in knowledge:
            if "name" in item.get("subj", "").lower() or "নাম" in item.get("subj", "") or "user_নাম" in item.get("subj", ""):
                return item.get("answer") or item.get("value")

    questions_list = [item.get("question", "") or item.get("subj", "") for item in knowledge]
    if not questions_list:
        return None

    try:
        all_texts = questions_list + [message]
        tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=True)
        matrix = tfidf.fit_transform(all_texts)
        
        query_vec = matrix[-1]
        item_vecs = matrix[:-1]
        similarities = cosine_similarity(query_vec, item_vecs)[0]
        
        if len(similarities) > 0:
            best_idx = similarities.argmax()
            if similarities[best_idx] > 0.15:  
                return knowledge[best_idx].get("answer") or knowledge[best_idx].get("value")
    except Exception:
        pass

    return None


# =====================================
# 5. GEMINI AI RESPONSE ENGINE
# =====================================

def get_gemini_response(message, chat_messages=None, file_paths=None):
    # প্রথমে লোকাল লার্নিং বা নলেজবেজ চেক করা
    knowledge_reply = search_knowledge_intelligent(message)
    if knowledge_reply and not file_paths:
        return knowledge_reply

    try:
        # চ্যাট হিস্ট্রি ফরম্যাট করা জেমিনির জন্য
        contents = []
        if chat_messages:
            for m in chat_messages[-10:]:
                role = "user" if m.get("role") == "user" else "model"
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=m.get("content", ""))]
                ))

        # আপলোড করা ফাইল বা ছবি জেমিনির কন্টেন্টে যুক্ত করা
        current_parts = []
        if file_paths:
            for f_path in file_paths:
                if os.path.exists(f_path):
                    # জেমিনি ফাইল আপলোড হ্যান্ডলার ব্যবহার করে ফাইল পাঠানো
                    uploaded_file = client.files.upload(file=f_path)
                    current_parts.append(uploaded_file)

        # মেসেজ টেক্সট যোগ করা
        current_parts.append(message)
        contents.append(contents.append(types.Content(role="user", parts=current_parts)) if current_parts else message)

        # Gemini 2.5 Flash বা উপযুক্ত ফ্রি মডেল কল করা
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=message if not file_paths else current_parts,
        )
        return response.text

    except Exception as e:
        # কোনো কারণে জেমিনি ফেইল করলে ফলব্যাক মেকানিজম
        user_vector = vectorizer.transform([message])
        probabilities = model.predict_proba(user_vector)[0]
        best_index = probabilities.argmax()
        confidence = probabilities[best_index]

        if confidence >= 0.30:
            intent = model.classes_[best_index]
            language = detect_language(message)
            return answers[intent][language]

        if detect_language(message) == "bn":
            return f"দুঃখিত! জেমিনি এপিআই কানেক্ট করা যায়নি। (ত্রুটি: {str(e)})"
        return f"Sorry! Gemini API connection failed. (Error: {str(e)})"


# =====================================
# 6. FLASK ROUTES
# =====================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"reply": "Please write a message."}), 400
    reply = get_gemini_response(message)
    return jsonify({"reply": reply})


@app.route("/learn", methods=["POST"])
def learn():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    answer = data.get("answer", "").strip()

    if not question or not answer:
        return jsonify({"success": False, "message": "Question and answer are required."}), 400

    knowledge = load_knowledge()
    knowledge.append({
        "id": str(uuid.uuid4()),
        "question": question,
        "answer": answer,
        "created_date": datetime.now().isoformat()
    })
    save_knowledge(knowledge)
    return jsonify({"success": True, "message": "AI learned successfully!"})


@app.route("/knowledge", methods=["GET"])
def knowledge_count():
    knowledge = load_knowledge()
    return jsonify({"total": len(knowledge)})


# =====================================
# 7. CHAT HISTORY & MULTIPLE FILES UPLOAD APIS
# =====================================

@app.route("/chats", methods=["POST"])
def create_chat():
    chats = load_chats()
    new_chat = {
        "id": str(uuid.uuid4()),
        "title": "New Chat",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages": []
    }
    chats.insert(0, new_chat)
    save_chats(chats)
    return jsonify({"success": True, "chat": new_chat})


@app.route("/chats", methods=["GET"])
def get_chats():
    chats = load_chats()
    result = []
    for chat in chats:
        result.append({
            "id": chat["id"],
            "title": chat["title"],
            "updated_at": chat["updated_at"]
        })
    return jsonify(result)


@app.route("/chats/<chat_id>", methods=["GET"])
def open_chat(chat_id):
    chats = load_chats()
    chat = find_chat(chats, chat_id)
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    return jsonify(chat)


@app.route("/chats/<chat_id>/message", methods=["POST"])
def save_chat_message(chat_id):
    user_message = request.form.get("message", "").strip()
    uploaded_files = request.files.getlist("files")

    file_names = []
    saved_file_paths = []
    
    for uploaded_file in uploaded_files:
        if uploaded_file and uploaded_file.filename:
            filename = secure_filename(uploaded_file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            uploaded_file.save(file_path)
            file_names.append(filename)
            saved_file_paths.append(file_path)

    if file_names and not user_message:
        user_message = f"দয়া করে এই ফাইলগুলো বিশ্লেষণ করো: {', '.join(file_names)}"

    chats = load_chats()
    chat = find_chat(chats, chat_id)
    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    learned_reply = extract_and_store_fact(user_message, chat_id) if not saved_file_paths else None
    if learned_reply:
        ai_reply = learned_reply
    else:
        ai_reply = get_gemini_response(user_message, chat.get("messages", []), saved_file_paths)

    now = datetime.now().isoformat()

    display_content = user_message
    if file_names:
        display_content += f" (📎 Attached Files: {', '.join(file_names)})"

    chat["messages"].append({"role": "user", "content": display_content, "time": now})
    chat["messages"].append({"role": "ai", "content": ai_reply, "time": now})

    if chat["title"] == "New Chat":
        chat["title"] = user_message[:35]
        if len(user_message) > 35:
            chat["title"] += "..."

    chat["updated_at"] = now
    chats.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    save_chats(chats)

    return jsonify({"success": True, "reply": ai_reply, "chat": chat})


@app.route("/chats/<chat_id>/rename", methods=["POST"])
def rename_chat(chat_id):
    data = request.get_json(silent=True) or {}
    title = data.get("title", "").strip()
    if not title or len(title) > 80:
        return jsonify({"error": "Title must be 1-80 characters"}), 400

    chats = load_chats()
    chat = find_chat(chats, chat_id)
    if not chat:
        return jsonify({"error": "Chat not found"}), 404

    chat["title"] = title
    chat["updated_at"] = datetime.now().isoformat()
    save_chats(chats)
    return jsonify({"success": True, "chat": chat})


@app.route("/chats/<chat_id>", methods=["DELETE"])
def delete_chat(chat_id):
    chats = load_chats()
    updated_chats = [chat for chat in chats if chat.get("id") != chat_id]
    if len(updated_chats) == len(chats):
        return jsonify({"error": "Chat not found"}), 404
    save_chats(updated_chats)
    return jsonify({"success": True, "message": "Chat deleted"})


# =====================================
# START SERVER
# =====================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)