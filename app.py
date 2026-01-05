from flask import Flask, request, jsonify, render_template
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
from model_loader import predict_text
import json

app = Flask(__name__)

# --- Connect to MySQL database ---
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="sentiment_app"
)
cursor = db.cursor(dictionary=True)

# --- Create users table if not exists ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fullname VARCHAR(255),
    email VARCHAR(255) UNIQUE,
    password VARCHAR(255)
)
""")

# --- Create sentiment_results table with proper columns ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS sentiment_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_email VARCHAR(255),
    text TEXT,
    tokens TEXT,
    vector LONGTEXT,
    prediction VARCHAR(50),
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
db.commit()

# --- Serve login page ---
@app.route("/")
def login_page():
    return render_template("login.html")

# --- Serve sentiment page ---
@app.route("/index.html")
def index_page():
    return render_template("index.html")

# --- Register endpoint ---
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    hashed_password = generate_password_hash(data.get("password"))

    try:
        cursor.execute(
            "INSERT INTO users (fullname, email, password) VALUES (%s, %s, %s)",
            (data.get("fullname"), data.get("email"), hashed_password)
        )
        db.commit()
        return jsonify({"status": "success", "message": "User registered successfully."})
    except mysql.connector.IntegrityError:
        return jsonify({"status": "error", "message": "Email already exists."})

# --- Login endpoint ---
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    cursor.execute("SELECT fullname, password FROM users WHERE email=%s", (data.get("email"),))
    user = cursor.fetchone()

    if user and check_password_hash(user["password"], data.get("password")):
        return jsonify({"status": "success", "fullname": user["fullname"]})
    return jsonify({"status": "error", "message": "Invalid email or password."})

# --- Sentiment prediction endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text")

    if not text:
        return jsonify({"prediction": "Neutral", "confidence": 0, "tokens": [], "vector": []})

    try:
        # Get prediction, tokens, vector
        prediction, confidence, tokens, vector = predict_text(text)
        
        print(f"📊 Prediction result:")
        print(f"   - Prediction: {prediction}")
        print(f"   - Confidence: {confidence}%")
        print(f"   - Tokens: {tokens}")
        print(f"   - Vector length: {len(vector)}")
        print(f"   - Active features: {sum(1 for v in vector if v != 0)}")

        user_email = data.get("email")

        # Store in database - convert lists to JSON strings
        if user_email:
            # Convert to JSON strings for storage
            tokens_json = json.dumps(tokens, ensure_ascii=False)
            vector_json = json.dumps(vector)
            
            cursor.execute(
                """INSERT INTO sentiment_results
                (user_email, text, tokens, vector, prediction, confidence)
                VALUES (%s, %s, %s, %s, %s, %s)""",
                (user_email, text, tokens_json, vector_json, prediction, confidence)
            )
            db.commit()
            print(f"✅ Saved to database for user: {user_email}")

        # Return response
        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "tokens": tokens,
            "vector": vector
        })
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "prediction": "Error", 
            "confidence": 0,
            "tokens": [],
            "vector": [],
            "error": str(e)
        }), 500

@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")

@app.route("/dashboard-data")
def dashboard_data():
    # Get user email from query parameter
    user_email = request.args.get('email')
    
    if not user_email:
        return jsonify({"error": "No user email provided"}), 400
    
    try:
        # Query only the logged-in user's data
        cursor.execute(
            "SELECT * FROM sentiment_results WHERE user_email = %s ORDER BY created_at DESC",
            (user_email,)
        )
        data = cursor.fetchall()
        
        print(f"📊 Retrieved {len(data)} records for user: {user_email}")
        
        # Process the data to make it display-friendly
        processed_data = []
        for row in data:
            # Debug print
            print(f"\n🔍 Processing row {row['id']}:")
            print(f"   Tokens raw: {row['tokens']}")
            print(f"   Vector raw (first 50 chars): {str(row['vector'])[:50]}")
            
            # Parse tokens and vector if they're JSON strings
            tokens = []
            vector = []
            
            try:
                if row['tokens']:
                    tokens = json.loads(row['tokens'])
                    print(f"   ✅ Parsed {len(tokens)} tokens: {tokens}")
            except Exception as e:
                print(f"   ❌ Error parsing tokens: {e}")
                tokens = []
            
            try:
                if row['vector']:
                    vector = json.loads(row['vector'])
                    print(f"   ✅ Parsed vector with {len(vector)} dimensions")
            except Exception as e:
                print(f"   ❌ Error parsing vector: {e}")
                vector = []
            
            # Count non-zero elements in vector for summary
            non_zero_count = sum(1 for v in vector if v != 0) if vector else 0
            
            processed_data.append({
                'id': row['id'],
                'user_email': row['user_email'],
                'text': row['text'],
                'tokens': tokens,
                'token_count': len(tokens),
                'vector_summary': f"{non_zero_count} active features out of {len(vector)}" if vector else "No vector data",
                'vector': vector,  # Keep full vector but won't display it
                'prediction': row['prediction'],
                'confidence': row['confidence'],
                'created_at': row['created_at'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(row['created_at'], 'strftime') else str(row['created_at'])
            })
        
        return jsonify(processed_data)
        
    except Exception as e:
        print(f"❌ Error in dashboard_data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
# --- Serve Model Details page ---
@app.route("/model-details")
def model_details_page():
    return render_template("model_details.html")

if __name__ == "__main__":
    app.run(debug=True)