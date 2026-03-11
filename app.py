from flask import Flask, render_template, request, redirect, session
import pickle
import sqlite3
import os

app = Flask(__name__)
app.secret_key = "secretkey"

# Check if model files exist
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    print("Model files missing! Run the training script first.")
    exit()

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Initialize database
def init_db():
    conn = sqlite3.connect("news.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        password TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        news TEXT,
        prediction TEXT,
        confidence REAL
    )""")
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        conn = sqlite3.connect("news.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username,password))
        user = c.fetchone()
        conn.close()
        if user:
            session["user"] = username
            return redirect("/")
        else:
            return "Invalid Credentials"
    return render_template("login.html")

@app.route("/register", methods=["POST"])
def register():
    username = request.form["username"]
    password = request.form["password"]
    conn = sqlite3.connect("news.db")
    c = conn.cursor()
    c.execute("INSERT INTO users(username,password) VALUES(?,?)",(username,password))
    conn.commit()
    conn.close()
    return redirect("/login")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    vector = vectorizer.transform([news])
    prediction = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0]
    confidence = max(prob)*100
    result = "Real News" if prediction==1 else "Fake News"

    # Save to history
    conn = sqlite3.connect("news.db")
    c = conn.cursor()
    c.execute("INSERT INTO history(news,prediction,confidence) VALUES(?,?,?)",
              (news,result,confidence))
    conn.commit()
    conn.close()

    return render_template("result.html", prediction=result, confidence=round(confidence,2), news=news)

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("news.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM history")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM history WHERE prediction='Real News'")
    real = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM history WHERE prediction='Fake News'")
    fake = c.fetchone()[0]
    conn.close()
    return render_template("dashboard.html", total=total, real=real, fake=fake)

@app.route("/history")
def history():
    conn = sqlite3.connect("news.db")
    c = conn.cursor()
    c.execute("SELECT news,prediction,confidence FROM history")
    data = c.fetchall()
    conn.close()
    return render_template("history.html", data=data)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")

if __name__=="__main__":
    app.run(debug=True)