import torch
import json
import torch.nn as nn
import re

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# Load vocab and labels
with open('vectorizer_vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

with open('label_map.json', 'r', encoding='utf-8') as f:
    label_map = json.load(f)

input_dim = len(vocab)
print("✅ Vocabulary size:", input_dim)
num_classes = len(label_map)

# Load model
model = LogisticRegression(input_dim, num_classes)
state_dict = torch.load('sentiment_model.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# --- Tokenize Nepali text properly ---
def tokenize_nepali(text):
    """
    Tokenize Nepali text by splitting on whitespace and punctuation
    """
    # Remove extra whitespace
    text = text.strip()
    
    # Split by whitespace first
    tokens = text.split()
    
    # Further split by punctuation if needed
    final_tokens = []
    for token in tokens:
        # Remove punctuation from start and end
        token = token.strip('.,!?;:()[]{}"\'-')
        if token:  # Only add non-empty tokens
            final_tokens.append(token)
    
    return final_tokens


def vectorize_text(text):
    """
    Convert text to TF-IDF vector representation
    Returns: tensor, tokens list, vector as list
    """
  
    tokens = tokenize_nepali(text)
    
    vector = torch.zeros(len(vocab), dtype=torch.float32)
    

    for word in tokens:
        if word in vocab:
            vector[vocab[word]] = 1.0
    
  
    vector_list = vector.tolist()
    
    return vector.unsqueeze(0), tokens, vector_list

def predict_text(text):
    """
    Predict sentiment of given text
    Returns: prediction, confidence, tokens, vector
    """
    
    x, tokens, vector = vectorize_text(text)
    
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100
        prediction = label_map[str(pred_idx)]
    
    print(f"✅ Prediction: {prediction} ({confidence:.1f}%)")
    print(f"✅ Tokens found: {len(tokens)} tokens")
    print(f"✅ Active features: {sum(1 for v in vector if v != 0)} out of {len(vector)}")
    
    return prediction, round(confidence, 1), tokens, vector


# --- Test function (optional) ---
def test_prediction():
    """Test the prediction function"""
    test_text = "यो राम्रो छ"
    print(f"\n🧪 Testing with: '{test_text}'")
    pred, conf, tokens, vec = predict_text(test_text)
    print(f"Tokens: {tokens}")
    print(f"Vector summary: {sum(1 for v in vec if v != 0)} active features")
    

if __name__ == "__main__":
    test_prediction()