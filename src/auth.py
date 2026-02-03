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

    def register_user(self, username, password):
        if username in self.users:
            return False, "Username already exists"
        
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        self.users[username] = hashed.decode('utf-8')
        self._save_users()
        return True, "User created successfully"

    def login_user(self, username, password):
        if username not in self.users:
            return False
        
        stored_hash = self.users[username].encode('utf-8')
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
            return True
        return False
