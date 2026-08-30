import streamlit as st
import json
import os
import hashlib

USERS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'users.json')

def init_users_file():
    if not os.path.exists(os.path.dirname(USERS_FILE)):
        os.makedirs(os.path.dirname(USERS_FILE))
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            # Default admin user
            default_users = {
                "admin": {
                    "password": hash_password("admin123"),
                    "name": "Administrator",
                    "email": "admin@example.com"
                }
            }
            json.dump(default_users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    init_users_file()
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def register_user(username, password, name, email):
    users = load_users()
    if username in users:
        return False, "Username already exists."
    
    users[username] = {
        "password": hash_password(password),
        "name": name,
        "email": email
    }
    save_users(users)
    return True, "Registration successful! Please login."

def authenticate_user(username, password):
    users = load_users()
    if username in users:
        if users[username]["password"] == hash_password(password):
            return True, users[username]
    return False, None

def render_login():
    st.markdown("<h2 style='text-align: center; color: #1e3a8a;'>Welcome Back</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>Sign in to continue to Election NLP</p>", unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign In")
        
        if submit:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                success, user_data = authenticate_user(username, password)
                if success:
                    st.session_state['authenticated'] = True
                    user_data['username'] = username  # store username key
                    st.session_state['current_user'] = user_data
                    st.session_state['page'] = 'Dashboard'
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
                    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back to Home", use_container_width=True):
            st.session_state['page'] = 'Landing'
            st.rerun()
    with col2:
        if st.button("Create Account", type="primary", use_container_width=True):
            st.session_state['page'] = 'Signup'
            st.rerun()

def render_signup():
    st.markdown("<h2 style='text-align: center; color: #1e3a8a;'>Create Your Account</h2>", unsafe_allow_html=True)
    
    with st.form("signup_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        submit = st.form_submit_button("Create Account")
        
        if submit:
            if not all([name, email, username, password, confirm_password]):
                st.error("Please fill in all fields.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                success, msg = register_user(username, password, name, email)
                if success:
                    st.success(msg)
                    st.session_state['page'] = 'Login'
                else:
                    st.error(msg)
                    
    if st.button("Already have an account? Sign In", use_container_width=True):
        st.session_state['page'] = 'Login'
        st.rerun()

def check_auth():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    return st.session_state['authenticated']

def logout():
    st.session_state['authenticated'] = False
    st.session_state['current_user'] = None
    st.session_state['page'] = 'Landing'
    st.rerun()
