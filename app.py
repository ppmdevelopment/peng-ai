from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

app = Flask(__name__)

# ============================================
# DATEIEN
# ============================================
LERN_DATEI = "peng_ai_lerndaten.json"
UNBEKANNT_DATEI = "peng_ai_unbekannt.json"

def lade_lerndaten():
    if os.path.exists(LERN_DATEI):
        with open(LERN_DATEI, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def speichere_lerndaten(daten):
    with open(LERN_DATEI, "w", encoding="utf-8") as f:
        json.dump(daten, f, ensure_ascii=False, indent=2)

def speichere_unbekannte_frage(frage):
    unbekannte = []
    if os.path.exists(UNBEKANNT_DATEI):
        with open(UNBEKANNT_DATEI, "r", encoding="utf-8") as f:
            unbekannte = json.load(f)
    if frage not in unbekannte:
        unbekannte.append(frage)
        with open(UNBEKANNT_DATEI, "w", encoding="utf-8") as f:
            json.dump(unbekannte, f, ensure_ascii=False, indent=2)

# ============================================
# TRAINING DATEN
# ============================================
basis_training = [
    ("ich bin anfänger", "anfänger"),
    ("wie fange ich an", "anfänger"),
    ("weiqi für neulinge", "anfänger"),
    ("ich kenne die regeln nicht", "anfänger"),
    ("wie spielt man weiqi", "anfänger"),
    ("erkläre mir weiqi", "anfänger"),
    ("ich bin neu", "anfänger"),
    ("was ist atari", "atari"),
    ("was bedeutet atari", "atari"),
    ("atari erklären", "atari"),
    ("wann ist ein stein in atari", "atari"),
    ("mein stein ist in gefahr", "atari"),
    ("was sind freiheiten", "freiheiten"),
    ("was sind liberties", "freiheiten"),
    ("wie viele freiheiten hat ein stein", "freiheiten"),
    ("freiheiten erklären", "freiheiten"),
    ("wie tötet man steine", "captures"),
    ("wie schlägt man steine", "captures"),
    ("wie killt man steine", "captures"),
    ("steine gefangen nehmen", "captures"),
    ("wie nimmt man steine vom brett", "captures"),
    ("wann kann ich steine nehmen", "captures"),
    ("wie fange ich gegnerische steine", "captures"),
    ("wie starte ich das spiel", "opening"),
    ("was ist fuseki", "opening"),
    ("opening strategie", "opening"),
    ("welche züge am anfang", "opening"),
    ("erste züge im spiel", "opening"),
    ("warum ecken zuerst", "ecken"),
    ("ecken strategie", "ecken"),
    ("sind ecken wichtig", "ecken"),
    ("warum spielt man in die ecken", "ecken"),
    ("was ist eine gruppe", "gruppen"),
    ("wie verbinde ich steine", "gruppen"),
    ("wann sind steine verbunden", "gruppen"),
    ("was bedeutet leben", "leben_tod"),
    ("zwei augen regel", "leben_tod"),
    ("wie überleben meine steine", "leben_tod"),
    ("wann ist eine gruppe tot", "leben_tod"),
    ("wann lebt eine gruppe", "leben_tod"),
    ("was ist ko", "ko"),
    ("ko regel erklären", "ko"),
    ("was bedeutet ko kampf", "ko"),
    ("wie gewinnt man", "punkte"),
    ("wie kann ich gewinnen", "punkte"),
    ("wie zählt man punkte", "punkte"),
    ("wer gewinnt", "punkte"),
    ("wie funktioniert die wertung", "punkte"),
    ("wann habe ich gewonnen", "punkte"),
    ("wie endet das spiel", "punkte"),
    ("gewinnbedingung", "punkte"),
    ("was ist territorium", "territorium"),
    ("wie baut man territorium", "territorium"),
    ("wie sichere ich gebiete", "territorium"),
    ("was ist joseki", "joseki"),
    ("joseki erklären", "joseki"),
    ("was ist tesuji", "tesuji"),
    ("tesuji erklären", "tesuji"),
    ("clevere züge", "tesuji"),
    ("was ist sente", "sente"),
    ("sente erklären", "sente"),
    ("initiative im spiel", "sente"),
    ("hallo", "begrüßung"),
    ("hi", "begrüßung"),
    ("hey", "begrüßung"),
]

antworten = {
    "anfänger": "🟢 Willkommen bei Weiqi! Zwei Spieler (Schwarz & Weiß), Schwarz beginnt, Steine auf Kreuzungspunkte setzen. Ziel: Mehr Territorium als der Gegner. Tipp: Lerne zuerst Atari zu erkennen!",
    "atari": "⚠️ Atari = ein Stein hat nur noch EINE Freiheit! Wie 'Schach' beim Schach. Wenn du nichts tust werden deine Steine geschlagen. Entkommen: neue Freiheit hinzufügen oder Gegendrohung spielen!",
    "freiheiten": "🔵 Freiheiten sind leere Felder direkt neben einem Stein. Mitte: 4 | Rand: 3 | Ecke: 2. Verbundene Steine teilen Freiheiten. Null Freiheiten = Stein wird geschlagen!",
    "captures": "⚔️ Besetze ALLE Freiheiten eines Steins → er wird vom Brett genommen. Gefangene Steine = Minuspunkte für den Gegner. Tipp: Erst Atari drohen, dann schlagen!",
    "opening": "🎯 Ecken zuerst → Seiten → Mitte. Spiele nicht zu früh Kämpfe. Beliebte erste Züge: 3-4, 4-4. Denke groß — baue Einfluss!",
    "ecken": "📐 In der Ecke nur 2 Seiten sichern statt 4 → effizientestes Territorium. Klassisch: Ecke → Seite → Zentrum.",
    "gruppen": "🔗 Berührende Steine sind verbunden und teilen Freiheiten. Starke Gruppen sind schwer zu schlagen. Trenne die Steine des Gegners!",
    "leben_tod": "❤️ Eine Gruppe lebt mit 2 echten Augen. Auge = leeres Feld komplett von deinen Steinen umgeben. Merke: 'Two eyes live, one eye dies'",
    "ko": "♾️ Ko = beide könnten denselben Stein ewig schlagen. Nach Ko-Schlag erst woanders spielen (Ko-Drohung). Ko-Kämpfe sind sehr strategisch!",
    "punkte": "🏆 Territorium = leere Felder von deinen Steinen umgeben. Gefangene Steine = Minuspunkte. Weiß bekommt Komi (6.5 Punkte). Wer mehr Punkte hat gewinnt!",
    "territorium": "🗺️ Leere Felder komplett von deinen Steinen umgeben = dein Territorium. Ecken am einfachsten. Zu offen = Gegner kann eindringen!",
    "joseki": "📚 Joseki = bekannte optimale Zügefolgen in den Ecken. Beide spielen optimal, Ergebnis ausgeglichen. Als Anfänger: lerne 3-4 einfache Joseki!",
    "tesuji": "✨ Tesuji = clevere taktische Züge die eine Situation lösen. Bekannt: Snapback, Ladder, Net. Übung: täglich Tsumego lösen!",
    "sente": "⚡ Sente = Gegner muss antworten → du hast Initiative. Gote = du verlierst Initiative. Sente-Züge sind wertvoller. 'Sente gewinnt Spiele'",
    "begrüßung": "👋 Hallo! Ich bin Peng AI, dein Weiqi-Assistent. Frag mich über Regeln, Strategien oder Züge!",
}

# ============================================
# MODELL
# ============================================
def trainiere_modell(training_daten):
    texte = [t for t, _ in training_daten]
    labels = [l for _, l in training_daten]
    modell = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    modell.fit(texte, labels)
    return modell

gespeicherte_daten = lade_lerndaten()
alle_training_daten = basis_training + gespeicherte_daten
modell = trainiere_modell(alle_training_daten)

# ============================================
# ROUTEN
# ============================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    global modell, gespeicherte_daten, alle_training_daten
    
    daten = request.json
    frage = daten.get("frage", "").lower().strip()
    
    if not frage:
        return jsonify({"antwort": "Bitte stelle eine Frage!"})
    
    intent = modell.predict([frage])[0]
    konfidenz = modell.predict_proba([frage]).max()
    
    if konfidenz < 0.2:
        speichere_unbekannte_frage(frage)
        return jsonify({
            "antwort": "Das weiß ich noch nicht — aber ich habe deine Frage gespeichert! 📝 ppm trainiert mich bald weiter.",
            "unbekannt": True
        })
    
    return jsonify({"antwort": antworten[intent]})

@app.route("/train", methods=["POST"])
def train():
    global modell, gespeicherte_daten, alle_training_daten
    
    daten = request.json
    passwort = daten.get("passwort", "")
    
    if passwort != "ppm-geheim":
        return jsonify({"erfolg": False, "fehler": "Falsches Passwort!"})
    
    frage = daten.get("frage", "").strip()
    intent = daten.get("intent", "").strip()
    
    # Ähnlichkeit prüfen
    thema_fragen = [t for t, l in alle_training_daten if l == intent]
    if thema_fragen:
        vectorizer = TfidfVectorizer()
        alle_vektoren = vectorizer.fit_transform(thema_fragen + [frage])
        ähnlichkeit = cosine_similarity(alle_vektoren[-1], alle_vektoren[:-1])[0].max()
        if ähnlichkeit < 0.15:
            return jsonify({
                "erfolg": False,
                "fehler": f"Frage passt nicht zum Thema! Ähnlichkeit nur {round(ähnlichkeit*100)}%"
            })
    
    neue_daten = gespeicherte_daten + [(frage, intent)]
    speichere_lerndaten(neue_daten)
    gespeicherte_daten = neue_daten
    alle_training_daten = basis_training + neue_daten
    modell = trainiere_modell(alle_training_daten)
    
    return jsonify({"erfolg": True, "beispiele": len(alle_training_daten)})

@app.route("/unbekannt", methods=["GET"])
def unbekannt():
    if os.path.exists(UNBEKANNT_DATEI):
        with open(UNBEKANNT_DATEI, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify([])

if __name__ == "__main__":
    app.run(debug=True)