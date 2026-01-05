import mysql.connector
import json

# Connect to database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="sentiment_app"
)
cursor = db.cursor(dictionary=True)

print("🔍 Checking sentiment_results table...")
print("=" * 80)

# Get all records
cursor.execute("SELECT * FROM sentiment_results ORDER BY created_at DESC LIMIT 5")
results = cursor.fetchall()

print(f"\n📊 Found {len(results)} recent records\n")

for i, row in enumerate(results, 1):
    print(f"\n{'='*80}")
    print(f"Record #{i} (ID: {row['id']})")
    print(f"{'='*80}")
    print(f"User: {row['user_email']}")
    print(f"Text: {row['text'][:100]}...")
    print(f"Prediction: {row['prediction']} ({row['confidence']}%)")
    print(f"Created: {row['created_at']}")
    
    print(f"\n📝 Tokens (raw):")
    print(f"   Type: {type(row['tokens'])}")
    print(f"   Value: {row['tokens']}")
    
    print(f"\n🔢 Vector (raw):")
    print(f"   Type: {type(row['vector'])}")
    if row['vector']:
        print(f"   Length: {len(str(row['vector']))} characters")
        print(f"   First 100 chars: {str(row['vector'])[:100]}")
    
    # Try to parse
    print(f"\n✅ Attempting to parse:")
    try:
        if row['tokens']:
            tokens = json.loads(row['tokens'])
            print(f"   Tokens parsed: {len(tokens)} tokens")
            print(f"   Tokens: {tokens}")
        else:
            print(f"   ❌ Tokens is None or empty")
    except Exception as e:
        print(f"   ❌ Error parsing tokens: {e}")
    
    try:
        if row['vector']:
            vector = json.loads(row['vector'])
            print(f"   Vector parsed: {len(vector)} dimensions")
            active = sum(1 for v in vector if v != 0)
            print(f"   Active features: {active}")
        else:
            print(f"   ❌ Vector is None or empty")
    except Exception as e:
        print(f"   ❌ Error parsing vector: {e}")

print(f"\n{'='*80}")
print("✅ Check complete!")

# Check table structure
print(f"\n\n📋 Table Structure:")
print("=" * 80)
cursor.execute("DESCRIBE sentiment_results")
columns = cursor.fetchall()
for col in columns:
    print(f"{col['Field']:20s} {col['Type']:20s} {col['Null']:5s} {col['Key']:5s}")

cursor.close()
db.close()