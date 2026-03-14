from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

app = Flask(__name__)

# ============================================
# FILES
# ============================================
LEARN_FILE = "peng_ai_learndata.json"
UNKNOWN_FILE = "peng_ai_unknown.json"

def load_learndata():
    if os.path.exists(LEARN_FILE):
        with open(LEARN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_learndata(data):
    with open(LEARN_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_unknown_question(question):
    unknown = []
    if os.path.exists(UNKNOWN_FILE):
        with open(UNKNOWN_FILE, "r", encoding="utf-8") as f:
            unknown = json.load(f)
    if question not in unknown:
        unknown.append(question)
        with open(UNKNOWN_FILE, "w", encoding="utf-8") as f:
            json.dump(unknown, f, ensure_ascii=False, indent=2)

# ============================================
# TRAINING DATA
# ============================================
base_training = [
    # Beginner
    ("i am a beginner", "beginner"),
    ("how do i start", "beginner"),
    ("go for beginners", "beginner"),
    ("i don't know the rules", "beginner"),
    ("how do you play go", "beginner"),
    ("explain go to me", "beginner"),
    ("i am new", "beginner"),
    ("what is go", "beginner"),
    ("teach me go", "beginner"),
    ("how to play weiqi", "beginner"),

    # Atari
    ("what is atari", "atari"),
    ("what does atari mean", "atari"),
    ("explain atari", "atari"),
    ("when is a stone in atari", "atari"),
    ("my stone is in danger", "atari"),
    ("atari in go", "atari"),

    # Liberties
    ("what are liberties", "liberties"),
    ("how many liberties does a stone have", "liberties"),
    ("explain liberties", "liberties"),
    ("what is a liberty", "liberties"),
    ("liberties in go", "liberties"),

    # Captures
    ("how do you capture stones", "captures"),
    ("how do you kill stones", "captures"),
    ("how to take stones", "captures"),
    ("how to remove stones from the board", "captures"),
    ("when can i capture", "captures"),
    ("how do i take enemy stones", "captures"),
    ("capturing stones", "captures"),

    # Opening
    ("how do i start the game", "opening"),
    ("what is fuseki", "opening"),
    ("opening strategy", "opening"),
    ("which moves at the beginning", "opening"),
    ("first moves in the game", "opening"),
    ("how to open in go", "opening"),

    # Corners
    ("why corners first", "corners"),
    ("corner strategy", "corners"),
    ("are corners important", "corners"),
    ("why play in the corners", "corners"),
    ("corners in go", "corners"),

    # Groups
    ("what is a group", "groups"),
    ("how do i connect stones", "groups"),
    ("when are stones connected", "groups"),
    ("stone groups", "groups"),
    ("connecting stones", "groups"),

    # Life and Death
    ("what does life mean", "life_death"),
    ("what does death mean", "life_death"),
    ("two eyes rule", "life_death"),
    ("how do my stones survive", "life_death"),
    ("when is a group dead", "life_death"),
    ("when does a group live", "life_death"),
    ("life and death in go", "life_death"),

    # Ko
    ("what is ko", "ko"),
    ("explain the ko rule", "ko"),
    ("what is a ko fight", "ko"),
    ("ko in go", "ko"),

    # Scoring
    ("how do you win", "scoring"),
    ("how do you count points", "scoring"),
    ("who wins", "scoring"),
    ("how does scoring work", "scoring"),
    ("what are points in go", "scoring"),
    ("how to win in go", "scoring"),
    ("winning in go", "scoring"),
    ("how do i win", "scoring"),

    # Territory
    ("what is territory", "territory"),
    ("how do you build territory", "territory"),
    ("explain territory", "territory"),
    ("how do i secure areas", "territory"),
    ("territory in go", "territory"),

    # Joseki
    ("what is joseki", "joseki"),
    ("explain joseki", "joseki"),
    ("joseki patterns", "joseki"),

    # Tesuji
    ("what is tesuji", "tesuji"),
    ("explain tesuji", "tesuji"),
    ("clever moves in go", "tesuji"),

    # Sente
    ("what is sente", "sente"),
    ("explain sente", "sente"),
    ("initiative in go", "sente"),
    ("what is gote", "sente"),

    # Greeting
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hey", "greeting"),
    ("good morning", "greeting"),
    ("sup", "greeting"),
]

answers = {
    "beginner": """🟢 Welcome to Go (Weiqi)! Here are the basics:
  • Two players: Black and White
  • Black always goes first
  • Stones are placed on intersections of the board
  • Goal: Surround more territory than your opponent
  • Tip: Learn to recognize Atari first — it's the most important skill!""",

    "atari": """⚠️ Atari means: A stone or group has only ONE liberty left!
  • Like "Check" in chess — a warning
  • If you do nothing, your stones will be captured
  • Escape by adding a new liberty or playing a counter-threat!""",

    "liberties": """🔵 Liberties are the empty points directly next to a stone:
  • Center stone: 4 liberties
  • Edge stone: 3 liberties
  • Corner stone: 2 liberties
  • Connected stones share their liberties
  • Zero liberties = stone gets captured!""",

    "captures": """⚔️ Capturing stones — here's how:
  • Fill ALL liberties of a stone or group
  • Those stones are removed from the board
  • Captured stones count as points against your opponent
  • Tip: Threaten with Atari first, then capture!""",

    "opening": """🎯 Opening (Fuseki):
  • Corners first → Sides → Center
  • Don't start fights too early
  • Popular first moves: 3-4, 4-4 points
  • Think big — build influence, not just small territories!""",

    "corners": """📐 Why corners first?
  • In a corner you only need to secure 2 sides instead of 4
  • Corners = most efficient territory
  • Classic order: Corner → Side → Center""",

    "groups": """🔗 Groups and connections:
  • Stones touching horizontally or vertically are connected
  • Connected stones share all their liberties
  • Strong groups are hard to capture
  • Try to connect your stones and separate your opponent's!""",

    "life_death": """❤️ Life and Death — the most important concept in Go:
  • A group LIVES if it has 2 real eyes
  • An eye = an empty point completely surrounded by your stones
  • With 2 eyes the opponent can never capture your group
  • Remember: 'Two eyes live, one eye dies'""",

    "ko": """♾️ Ko — the most exciting rule:
  • Ko happens when both players could capture the same stone forever
  • After a Ko capture, you CANNOT recapture immediately
  • You must play somewhere else first (Ko threat)
  • Ko fights are very strategic!""",

    "scoring": """🏆 How to win in Go:
  • Territory = empty points completely surrounded by your stones
  • Captured stones = penalty points for your opponent
  • White gets Komi (usually 6.5 points) to compensate for going second
  • Most points wins!""",

    "territory": """🗺️ Building territory:
  • Empty points completely surrounded by your stones = your territory
  • Corners are easiest to secure
  • Too open territory can be invaded by your opponent
  • Balance between attack and defense!""",

    "joseki": """📚 Joseki — established corner sequences:
  • Joseki are known optimal move sequences in the corners
  • Both players play optimally — the result is balanced
  • Hundreds of different joseki exist
  • As a beginner: learn 3-4 simple joseki
  • Important: joseki depend on the rest of the board!""",

    "tesuji": """✨ Tesuji — clever tactical moves:
  • Tesuji are elegant moves that solve a local situation
  • Famous tesuji: Snapback, Ladder (Shicho), Net (Geta)
  • A good tesuji can turn the game around
  • Practice: solve Tsumego (problems) every day!""",

    "sente": """⚡ Sente and Gote:
  • Sente = your move forces your opponent to respond → you keep the initiative
  • Gote = your opponent doesn't have to respond → you lose the initiative
  • Sente moves are more valuable!
  • 'Sente wins games'""",

    "greeting": "👋 Hello! I'm Peng AI, your Go (Weiqi) assistant. Ask me anything about rules, strategies, or moves — I'm here to help you improve! 围棋",
}

# ============================================
# MODEL
# ============================================
def train_model(training_data):
    texts = [t for t, _ in training_data]
    labels = [l for _, l in training_data]
    model = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    model.fit(texts, labels)
    return model

saved_data = load_learndata()
all_training_data = base_training + saved_data
model = train_model(all_training_data)

# ============================================
# ROUTES
# ============================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global model, saved_data, all_training_data

    data = request.json
    question = data.get("question", "").lower().strip()

    if not question:
        return jsonify({"answer": "Please ask a question!"})

    intent = model.predict([question])[0]
    confidence = model.predict_proba([question]).max()

    if confidence < 0.2:
        save_unknown_question(question)
        return jsonify({
            "answer": "I don't know that yet — but I've saved your question! 📝 ppm will train me soon.",
            "unknown": True
        })

    return jsonify({"answer": answers[intent]})

@app.route("/train", methods=["POST"])
def train():
    global model, saved_data, all_training_data

    data = request.json
    password = data.get("password", "")

    if password != "ppm-geheim":
        return jsonify({"success": False, "error": "Wrong password!"})

    question = data.get("question", "").strip()
    intent = data.get("intent", "").strip()

    # Check similarity
    topic_questions = [t for t, l in all_training_data if l == intent]
    if topic_questions:
        vectorizer = TfidfVectorizer()
        all_vectors = vectorizer.fit_transform(topic_questions + [question])
        similarity = cosine_similarity(all_vectors[-1], all_vectors[:-1])[0].max()
        if similarity < 0.15:
            return jsonify({
                "success": False,
                "error": f"Question doesn't match the topic! Similarity only {round(similarity*100)}%"
            })

    new_data = saved_data + [(question, intent)]
    save_learndata(new_data)
    saved_data = new_data
    all_training_data = base_training + new_data
    model = train_model(all_training_data)

    return jsonify({"success": True, "examples": len(all_training_data)})

@app.route("/unknown", methods=["GET"])
def unknown():
    if os.path.exists(UNKNOWN_FILE):
        with open(UNKNOWN_FILE, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify([])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)