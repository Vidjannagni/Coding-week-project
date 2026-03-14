"""Blueprint d'authentification, profil utilisateur, historique et administration."""

import os
import sqlite3
import logging

import bcrypt
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from config import (
    DB_PATH, MIN_USERNAME_LENGTH, MIN_PASSWORD_LENGTH,
    DEFAULT_ADMIN_USERNAME, DEFAULT_ADMIN_PASSWORD,
)

auth_bp = Blueprint("auth", __name__)
logger = logging.getLogger(__name__)


def get_db():
    """Open a SQLite connection configured to return dict-like rows."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


class User(UserMixin):
    def __init__(self, id, username, password_hash, is_admin=False):
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.is_admin = is_admin


def setup_login_manager(app):
    """Attach Flask-Login to the app and define how users are reloaded."""
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Veuillez vous connecter pour accéder à cette page."
    login_manager.login_message_category = "info"

    @login_manager.user_loader
    def load_user(user_id):
        conn = get_db()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        conn.close()
        if row:
            return User(row["id"], row["username"], row["password_hash"],
                        bool(row["is_admin"]))
        return None

    return login_manager


def init_db():
    """Create required tables, run lightweight migrations, and ensure an admin exists."""
    logger.info("DB initialization started | path=%s", DB_PATH)
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            age REAL,
            sex TEXT,
            prediction INTEGER,
            confidence REAL,
            probability REAL,
            form_data TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    # Backward-compatible schema updates.
    cursor = conn.execute("PRAGMA table_info(history)")
    columns = [col[1] for col in cursor.fetchall()]
    if "user_id" not in columns:
        conn.execute("ALTER TABLE history ADD COLUMN user_id INTEGER DEFAULT 0")
        logger.info("DB migration applied: history.user_id")
    if "patient_first_name" not in columns:
        conn.execute("ALTER TABLE history ADD COLUMN patient_first_name TEXT DEFAULT ''")
        logger.info("DB migration applied: history.patient_first_name")
    if "patient_last_name" not in columns:
        conn.execute("ALTER TABLE history ADD COLUMN patient_last_name TEXT DEFAULT ''")
        logger.info("DB migration applied: history.patient_last_name")

    # Users table migration for old databases.
    cursor_u = conn.execute("PRAGMA table_info(users)")
    user_columns = [col[1] for col in cursor_u.fetchall()]
    if "is_admin" not in user_columns:
        conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")
        logger.info("DB migration applied: users.is_admin")

    # Bootstrap a default admin account if none is present.
    admin = conn.execute("SELECT id FROM users WHERE is_admin = 1").fetchone()
    if admin is None:
        admin_pw = bcrypt.hashpw(DEFAULT_ADMIN_PASSWORD.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        try:
            conn.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 1)",
                         (DEFAULT_ADMIN_USERNAME, admin_pw))
            logger.info("Default admin account created")
        except sqlite3.IntegrityError:
            conn.execute("UPDATE users SET is_admin = 1 WHERE username = ?", (DEFAULT_ADMIN_USERNAME,))
            logger.info("Existing default admin promoted to admin role")

    conn.commit()
    conn.close()
    logger.info("DB initialization complete")


# --- Auth routes ---

@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        logger.info("Register attempt | username=%s", username)
        password = request.form.get("password", "")
        confirm = request.form.get("confirm", "")

        if not username or not password:
            flash("Nom d'utilisateur et mot de passe requis.", "error")
            return redirect(url_for("auth.register"))
        if len(username) < MIN_USERNAME_LENGTH:
            flash(f"Le nom d'utilisateur doit contenir au moins {MIN_USERNAME_LENGTH} caractères.", "error")
            return redirect(url_for("auth.register"))
        if len(password) < MIN_PASSWORD_LENGTH:
            flash(f"Le mot de passe doit contenir au moins {MIN_PASSWORD_LENGTH} caractères.", "error")
            return redirect(url_for("auth.register"))
        if password != confirm:
            flash("Les mots de passe ne correspondent pas.", "error")
            return redirect(url_for("auth.register"))

        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        conn = get_db()
        try:
            conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                         (username, pw_hash))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            logger.warning("Register rejected | username already exists: %s", username)
            flash("Ce nom d'utilisateur est déjà pris.", "error")
            return redirect(url_for("auth.register"))

        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        user = User(row["id"], row["username"], row["password_hash"], bool(row["is_admin"]))
        login_user(user)
        logger.info("Register success | user_id=%s | username=%s", user.id, user.username)
        flash(f"Bienvenue, {username} !", "success")
        return redirect(url_for("home"))

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        logger.info("Login attempt | username=%s", username)
        password = request.form.get("password", "")

        conn = get_db()
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if row and bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8")):
            user = User(row["id"], row["username"], row["password_hash"], bool(row["is_admin"]))
            login_user(user, remember=request.form.get("remember"))
            logger.info("Login success | user_id=%s | username=%s", user.id, user.username)
            flash(f"Bon retour, {username} !", "success")
            next_page = request.args.get("next")
            if next_page and not next_page.startswith("/"):
                next_page = None
            return redirect(next_page or url_for("home"))
        else:
            logger.warning("Login failed | username=%s", username)
            flash("Nom d'utilisateur ou mot de passe incorrect.", "error")
            return redirect(url_for("auth.login"))

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logger.info("Logout | user_id=%s", current_user.id)
    logout_user()
    flash("Vous avez été déconnecté.", "info")
    return redirect(url_for("home"))


# --- History routes ---

@auth_bp.route("/history")
@login_required
def history():
    conn = get_db()
    rows = conn.execute("SELECT * FROM history WHERE user_id = ? ORDER BY id DESC",
                        (current_user.id,)).fetchall()
    conn.close()
    records = [{**dict(r), "patient_first_name": r["patient_first_name"] or "", "patient_last_name": r["patient_last_name"] or ""} for r in rows]
    logger.info("History fetched | user_id=%s | records=%d", current_user.id, len(records))
    return render_template("history.html", records=records)


@auth_bp.route("/history/<int:record_id>", methods=["DELETE"])
@login_required
def delete_record(record_id):
    conn = get_db()
    conn.execute("DELETE FROM history WHERE id = ? AND user_id = ?",
                 (record_id, current_user.id))
    conn.commit()
    conn.close()
    logger.info("History record deleted | user_id=%s | record_id=%s", current_user.id, record_id)
    return jsonify({"status": "ok"})


@auth_bp.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    conn = get_db()
    conn.execute("DELETE FROM history WHERE user_id = ?", (current_user.id,))
    conn.commit()
    conn.close()
    logger.info("History cleared | user_id=%s", current_user.id)
    return jsonify({"status": "ok"})


# --- Profile route ---

@auth_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        new_username = request.form.get("username", "").strip()
        logger.info("Profile update attempt | user_id=%s", current_user.id)
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        conn = get_db()
        row = conn.execute("SELECT * FROM users WHERE id = ?", (current_user.id,)).fetchone()

        if not bcrypt.checkpw(current_password.encode("utf-8"), row["password_hash"].encode("utf-8")):
            conn.close()
            flash("Mot de passe actuel incorrect.", "error")
            return redirect(url_for("auth.profile"))

        if new_username and new_username != current_user.username:
            if len(new_username) < MIN_USERNAME_LENGTH:
                conn.close()
                flash(f"Le nom d'utilisateur doit contenir au moins {MIN_USERNAME_LENGTH} caractères.", "error")
                return redirect(url_for("auth.profile"))
            existing = conn.execute("SELECT id FROM users WHERE username = ? AND id != ?",
                                    (new_username, current_user.id)).fetchone()
            if existing:
                conn.close()
                flash("Ce nom d'utilisateur est déjà pris.", "error")
                return redirect(url_for("auth.profile"))
            conn.execute("UPDATE users SET username = ? WHERE id = ?",
                         (new_username, current_user.id))
            current_user.username = new_username

        if new_password:
            if len(new_password) < MIN_PASSWORD_LENGTH:
                conn.close()
                flash(f"Le nouveau mot de passe doit contenir au moins {MIN_PASSWORD_LENGTH} caractères.", "error")
                return redirect(url_for("auth.profile"))
            if new_password != confirm_password:
                conn.close()
                flash("Les mots de passe ne correspondent pas.", "error")
                return redirect(url_for("auth.profile"))
            pw_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                         (pw_hash, current_user.id))
            current_user.password_hash = pw_hash

        conn.commit()
        conn.close()
        flash("Profil mis à jour avec succès.", "success")
        logger.info("Profile updated | user_id=%s", current_user.id)
        return redirect(url_for("auth.profile"))

    conn = get_db()
    diag_count = conn.execute("SELECT COUNT(*) FROM history WHERE user_id = ?",
                              (current_user.id,)).fetchone()[0]
    conn.close()
    return render_template("profile.html", diag_count=diag_count)


# --- Admin routes ---

@auth_bp.route("/admin")
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash("Accès réservé aux administrateurs.", "error")
        return redirect(url_for("home"))

    conn = get_db()
    users = conn.execute("SELECT id, username, is_admin FROM users ORDER BY id").fetchall()
    total_diag = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    user_stats = []
    for u in users:
        count = conn.execute("SELECT COUNT(*) FROM history WHERE user_id = ?",
                             (u["id"],)).fetchone()[0]
        user_stats.append({
            "id": u["id"],
            "username": u["username"],
            "is_admin": bool(u["is_admin"]),
            "diag_count": count,
        })
    conn.close()
    logger.info("Admin dashboard opened | admin_user_id=%s | users=%d", current_user.id, len(user_stats))
    return render_template("admin.html", users=user_stats, total_diag=total_diag)


@auth_bp.route("/admin/toggle/<int:user_id>", methods=["POST"])
@login_required
def admin_toggle(user_id):
    if not current_user.is_admin:
        return jsonify({"error": "Non autorisé"}), 403
    if user_id == current_user.id:
        return jsonify({"error": "Impossible de modifier votre propre rôle"}), 400

    conn = get_db()
    row = conn.execute("SELECT is_admin FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        conn.close()
        return jsonify({"error": "Utilisateur introuvable"}), 404
    new_val = 0 if row["is_admin"] else 1
    conn.execute("UPDATE users SET is_admin = ? WHERE id = ?", (new_val, user_id))
    conn.commit()
    conn.close()
    logger.info("Admin role toggled | admin_user_id=%s | target_user_id=%s | is_admin=%s", current_user.id, user_id, bool(new_val))
    return jsonify({"status": "ok", "is_admin": bool(new_val)})


@auth_bp.route("/admin/delete/<int:user_id>", methods=["DELETE"])
@login_required
def admin_delete_user(user_id):
    if not current_user.is_admin:
        return jsonify({"error": "Non autorisé"}), 403
    if user_id == current_user.id:
        return jsonify({"error": "Impossible de supprimer votre propre compte"}), 400

    conn = get_db()
    conn.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    logger.info("User deleted by admin | admin_user_id=%s | target_user_id=%s", current_user.id, user_id)
    return jsonify({"status": "ok"})
