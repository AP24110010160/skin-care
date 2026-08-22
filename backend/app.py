import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

app = FastAPI(title="SkinSense AI Backend")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "168000080775-2aohtfi9hrk6v0naq27uj94rrnhpbpcb.apps.googleusercontent.com")

# Enable CORS for local testing if needed, though we will serve pages from this same port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Knowledge Base
KB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skincare_kb.json")
try:
    with open(KB_PATH, "r") as f:
        KB = json.load(f)
except Exception as e:
    # Fallback to local directory if relative directory resolver differs
    with open("skincare_kb.json", "r") as f:
        KB = json.load(f)

INGREDIENTS = KB["ingredients"]
EXPLICIT_PAIRS = KB["explicit_pairs"]

def find_ingredient(name: str):
    """Finds canonical name and ingredient data matching name or alias (case-insensitive)."""
    name_lower = name.strip().lower()
    for canonical_name, data in INGREDIENTS.items():
        if canonical_name.lower() == name_lower:
            return canonical_name, data
        for alias in data.get("aliases", []):
            if alias.lower() == name_lower:
                return canonical_name, data
    return None, None

@app.get("/api/ingredients")
def get_ingredients():
    """Returns a list of all ingredients with their canonical names and fields."""
    result = []
    for canonical_name, data in INGREDIENTS.items():
        item = {"name": canonical_name}
        item.update(data)
        result.append(item)
    return result

@app.get("/api/ingredients/{name}")
def get_ingredient(name: str):
    """Returns the full record for an ingredient with derived best_with, avoid_mixing, and related_ingredients."""
    canonical_name, data = find_ingredient(name)
    if not canonical_name:
        raise HTTPException(status_code=404, detail=f"Ingredient '{name}' not found")

    best_with = []
    avoid_mixing = []

    # Compute best_with and avoid_mixing from explicit_pairs
    for pair_entry in EXPLICIT_PAIRS:
        pair = pair_entry["pair"]
        if canonical_name in pair:
            # Find the other ingredient in the pair
            other_name = pair[0] if pair[1] == canonical_name else pair[1]
            relationship = pair_entry["relationship"]
            note = pair_entry["note"]

            if relationship == "safe_together":
                best_with.append({"name": other_name, "note": note, "relationship": relationship})
            else:
                avoid_mixing.append({"name": other_name, "note": note, "relationship": relationship})

    # Compute related ingredients:
    # 1. From explicit pairs involving this ingredient
    related_names = []
    for item in best_with + avoid_mixing:
        if item["name"] not in related_names:
            related_names.append(item["name"])

    # 2. Fallback to same conflict_class if < 3 related ingredients
    current_class = data.get("conflict_class")
    if len(related_names) < 3 and current_class:
        for other_name, other_data in INGREDIENTS.items():
            if other_name != canonical_name and other_data.get("conflict_class") == current_class:
                if other_name not in related_names:
                    related_names.append(other_name)
                    if len(related_names) >= 3:
                        break

    # 3. Fallback to any other ingredients if still < 3
    if len(related_names) < 3:
        for other_name in INGREDIENTS.keys():
            if other_name != canonical_name and other_name not in related_names:
                related_names.append(other_name)
                if len(related_names) >= 3:
                    break

    # Construct the related ingredients records
    related_records = []
    for r_name in related_names[:5]: # limit to top 5 related ingredients
        r_data = INGREDIENTS.get(r_name, {})
        related_records.append({
            "name": r_name,
            "does": r_data.get("does", ""),
            "helps": r_data.get("helps", []),
            "when": r_data.get("when", ""),
            "conflict_class": r_data.get("conflict_class", "")
        })

    response_data = {}
    response_data.update(data)
    response_data.update({
        "name": canonical_name,
        "best_with": best_with,
        "avoid_mixing": avoid_mixing,
        "related_ingredients": related_records
    })
    return response_data

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
from validate_response import validate_response

SYSTEM_PROMPT = """You are a skincare guidance assistant.
- Answer user questions DIRECTLY and ACTIONABLY in the very first sentence.
- Provide clear routine guidance (e.g. alternate nights, morning vs. night).
- Do not list long general ingredient definitions unless explicitly asked.
- Keep medical disclaimers to a single brief sentence at the end.
- You are an educational guide, not a dermatologist."""

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class GoogleAuthPayload(BaseModel):
    credential: str

@app.post("/api/chat")
def chat_endpoint(payload: ChatRequest):
    # Prepend system prompt
    formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in payload.messages:
        formatted_messages.append({"role": msg.role, "content": msg.content})

    # Forward to mlx_lm.server
    try:
        response = requests.post(
            "http://localhost:8080/v1/chat/completions",
            json={
                "messages": formatted_messages,
                "temperature": 0.7,
                "repetition_penalty": 1.3,
                "max_tokens": 512
            },
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        resp_json = response.json()
        reply = resp_json["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with local LLM: {str(e)}")

    # Extract last user message for validation
    last_user_message = ""
    for msg in reversed(payload.messages):
        if msg.role == "user":
            last_user_message = msg.content
            break

    # Validate response
    warning = None
    if last_user_message:
        validation = validate_response(last_user_message, reply)
        if not validation["ok"]:
            warning = "\n".join(validation["warnings"])

    return {"reply": reply, "warning": warning}

@app.post("/api/auth/google")
def verify_google_token(payload: GoogleAuthPayload):
    try:
        # Verify the token with Google
        id_info = id_token.verify_oauth2_token(
            payload.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        email = id_info.get("email")
        name = id_info.get("name")
        picture = id_info.get("picture")

        # In a real app, you would look up/create the user in your DB here
        # and generate a session token / JWT.
        
        return {
            "success": True,
            "user": {
                "email": email,
                "name": name,
                "picture": picture
            },
            "token": "dummy-session-token-for-dev"
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google ID token: {str(e)}")

# Let's add basic routes to serve pages if they are copied to frontend/
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")


@app.get("/")
def read_home():
    home_path = os.path.join(FRONTEND_DIR, "home.html")
    if os.path.exists(home_path):
        return FileResponse(home_path)
    return {"message": "Home page not copied yet"}

@app.get("/ingredients")
def read_ingredients():
    ingredients_path = os.path.join(FRONTEND_DIR, "ingredients.html")
    if os.path.exists(ingredients_path):
        return FileResponse(ingredients_path)
    return {"message": "Ingredients page not copied yet"}

@app.get("/chat")
def read_chat():
    chat_path = os.path.join(FRONTEND_DIR, "chat.html")
    if os.path.exists(chat_path):
        return FileResponse(chat_path)
    return {"message": "Chat page not copied yet"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
