import streamlit as st
import psycopg2
from db_operations import db_manager
import hashlib
import re
import bcrypt

def init_auth_tables():
    """Initialise les tables d'authentification"""
    commands = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            full_name VARCHAR(100),
            role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('user', 'admin')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS user_sessions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            session_token VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
        "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token)"
    ]
    
    try:
        conn = db_manager._get_connection()
        with conn.cursor() as cursor:
            for command in commands:
                cursor.execute(command)
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation des tables d'authentification: {e}")
        return False

def hash_password(password):
    """Hash un mot de passe avec bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def validate_email(email):
    """Valide le format d'email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def register_user(username, email, password, full_name):
    """Enregistre un nouvel utilisateur avec bcrypt"""
    if not validate_email(email):
        return False, "Format d'email invalide"
    
    if len(password) < 6:
        return False, "Le mot de passe doit contenir au moins 6 caractères"
    
    # Utiliser bcrypt au lieu de SHA-256
    password_hash = hash_password(password)
    
    try:
        conn = db_manager._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, full_name) VALUES (%s, %s, %s, %s)",
                (username, email, password_hash, full_name)
            )
        conn.commit()
        return True, "Utilisateur créé avec succès"
    except psycopg2.IntegrityError:
        return False, "Le nom d'utilisateur ou l'email existe déjà"
    except Exception as e:
        return False, f"Erreur lors de la création de l'utilisateur: {e}"
def authenticate_user(username, password):
    """Authentifie un utilisateur et récupère toutes les informations"""
    try:
        conn = db_manager._get_connection()
        with conn.cursor() as cursor:
            cursor.execute(
                """SELECT id, username, full_name, email, phone_number, department, role, password_hash 
                   FROM users WHERE username = %s AND is_active = TRUE""",
                (username,)
            )
            user = cursor.fetchone()
        
        if user and verify_password(password, user[7]):  # user[7] = password_hash
            return True, {
                "id": user[0],
                "username": user[1],
                "full_name": user[2],
                "email": user[3],
                "phone_number": user[4],
                "department": user[5],
                "role": user[6]
            }
        else:
            return False, "Nom d'utilisateur ou mot de passe incorrect"
    except Exception as e:
        return False, f"Erreur d'authentification: {e}"
def show_login_register():
    """Affiche les formulaires de login et register"""
    tab1, tab2 = st.tabs(["Connexion", "Inscription"])
    
    with tab1:
        st.subheader("Connexion")
        username = st.text_input("Nom d'utilisateur", key="login_username")
        password = st.text_input("Mot de passe", type="password", key="login_password")
        
        if st.button("Se connecter", key="login_btn"):
            if username and password:
                success, result = authenticate_user(username, password)
                if success:
                    st.session_state.user = result
                    st.session_state.logged_in = True
                    st.success(f"Bienvenue {result['full_name']}!")
                    st.rerun()
                else:
                    st.error(result)
            else:
                st.warning("Veuillez remplir tous les champs")
    
    with tab2:
        st.subheader("Inscription")
        full_name = st.text_input("Nom complet", key="register_name")
        email = st.text_input("Email", key="register_email")
        username = st.text_input("Nom d'utilisateur", key="register_username")
        password = st.text_input("Mot de passe", type="password", key="register_password")
        confirm_password = st.text_input("Confirmer le mot de passe", type="password", key="register_confirm")
        
        if st.button("Créer un compte", key="register_btn"):
            if not all([full_name, email, username, password, confirm_password]):
                st.warning("Veuillez remplir tous les champs")
            elif password != confirm_password:
                st.error("Les mots de passe ne correspondent pas")
            else:
                success, result = register_user(username, email, password, full_name)
                if success:
                    st.success(result)
                    st.info("Vous pouvez maintenant vous connecter")
                else:
                    st.error(result)

def check_authentication():
    """Vérifie si l'utilisateur est authentifié"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user = None
    
    # Initialiser les tables d'authentification
    if not st.session_state.logged_in:
        init_auth_tables()
    
    return st.session_state.logged_in

def logout():
    """Déconnecte l'utilisateur"""
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()
    
    
def verify_password(plain_password, hashed_password):
    """Vérifie si un mot de passe correspond au hash bcrypt"""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False