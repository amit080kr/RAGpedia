import json
import bcrypt
import os

USER_DB_FILE = "users.json"

class AuthManager:
    def __init__(self):
        self.users = self._load_users()

    def _load_users(self):
        if not os.path.exists(USER_DB_FILE):
            return {}
        try:
            with open(USER_DB_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _save_users(self):
        with open(USER_DB_FILE, "w") as f:
            json.dump(self.users, f)

    def register_user(self, username, password, secret_question, secret_answer):
        if username in self.users:
            return False, "Username already exists"
        
        hashed_pass = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        hashed_answer = bcrypt.hashpw(secret_answer.lower().strip().encode('utf-8'), bcrypt.gensalt())
        
        self.users[username] = {
            "password": hashed_pass.decode('utf-8'),
            "secret_question": secret_question,
            "secret_answer": hashed_answer.decode('utf-8')
        }
        self._save_users()
        return True, "User created successfully"

    def login_user(self, username, password):
        if username not in self.users:
            return False
        
        user_data = self.users[username]
        # Handle old format if it exists (just string hash)
        stored_hash = user_data["password"] if isinstance(user_data, dict) else user_data
        
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return True
        return False

    def get_user_question(self, username):
        if username in self.users and isinstance(self.users[username], dict):
            return self.users[username].get("secret_question")
        return None

    def reset_password(self, username, secret_answer, new_password):
        if username not in self.users:
            return False, "User not found"
        
        user_data = self.users[username]
        if not isinstance(user_data, dict):
            return False, "Legacy user: cannot reset password without secret question"
            
        stored_answer_hash = user_data["secret_answer"].encode('utf-8')
        if bcrypt.checkpw(secret_answer.lower().strip().encode('utf-8'), stored_answer_hash):
            new_hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            self.users[username]["password"] = new_hashed.decode('utf-8')
            self._save_users()
            return True, "Password reset successfully"
        
        return False, "Incorrect answer to secret question"
