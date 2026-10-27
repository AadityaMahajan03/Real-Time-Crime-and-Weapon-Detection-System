"""Role-Based Access Control (RBAC) and authentication service with Argon2 password hashing."""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Set

import bcrypt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from src.fasttrack.core.clock import Clock, SystemClock
from src.fasttrack.core.errors import FastTrackError
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import User, UserRole
from src.fasttrack.storage.db import DatabaseManager

logger = get_logger("auth.service")

# Permission sets per role
ROLE_PERMISSIONS: Dict[UserRole, Set[str]] = {
    UserRole.VIEWER: {
        "view_incidents",
        "view_streams",
        "view_audit",
        "view_forensics",
    },
    UserRole.REVIEWER: {
        "view_incidents",
        "view_streams",
        "view_audit",
        "view_forensics",
        "mark_reviewed",
        "mark_false_alarm",
        "escalate",
        "add_note",
    },
    UserRole.ADMIN: {
        "view_incidents",
        "view_streams",
        "view_audit",
        "view_forensics",
        "mark_reviewed",
        "mark_false_alarm",
        "escalate",
        "add_note",
        "manage_cameras",
        "manage_users",
        "export_audit",
        "system_settings",
    },
}

_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,  # 64 MB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


def hash_password(plain_password: str) -> str:
    """Hash password string using Argon2id with high computational hardness."""
    if not plain_password:
        raise ValueError("Password cannot be empty.")
    return _hasher.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against hashed string, supporting Argon2id and legacy bcrypt."""
    if not plain_password or not hashed_password:
        return False

    # Check for Argon2 hash
    if hashed_password.startswith("$argon2"):
        try:
            return _hasher.verify(hashed_password, plain_password)
        except (VerifyMismatchError, InvalidHashError):
            return False
        except Exception as e:
            logger.warning("Argon2 verification unexpected error: %s", e)
            return False

    # Check for Bcrypt hash
    if hashed_password.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
        except Exception as e:
            logger.warning("Bcrypt verification error: %s", e)
            return False

    return False


def can_user_perform(role: UserRole, action: str) -> bool:
    """Evaluate whether user role is authorized to perform the requested action."""
    allowed = ROLE_PERMISSIONS.get(role, set())
    return action in allowed


class AuthService:
    """Authentication and user management service backed by SQLite."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        clock: Optional[Clock] = None,
    ) -> None:
        self.db = db_manager
        self.clock = clock or SystemClock()

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Authenticate user by username and password. Updates last_login_utc on success."""
        clean_user = username.strip().lower()
        rows = self.db.execute_query(
            "SELECT id, username, password_hash, role, active, last_login_utc FROM users WHERE lower(username) = ?",
            (clean_user,),
        )
        if not rows:
            logger.warning("Authentication failed: user %s not found.", clean_user)
            return None

        row = rows[0]
        if not bool(row["active"]):
            logger.warning("Authentication failed: user %s is disabled.", clean_user)
            return None

        if not verify_password(password, row["password_hash"]):
            logger.warning("Authentication failed: invalid credentials for %s.", clean_user)
            return None

        now_iso = self.clock.now_utc().isoformat()
        try:
            self.db.execute_non_query(
                "UPDATE users SET last_login_utc = ? WHERE id = ?",
                (now_iso, row["id"]),
            )
        except Exception as e:
            logger.warning("Failed to record last login for %s: %s", clean_user, e)

        role_enum = UserRole(row["role"]) if row["role"] in UserRole.__members__.values() else UserRole.VIEWER
        return User(
            id=row["id"],
            username=row["username"],
            password_hash=row["password_hash"],
            role=role_enum,
            active=bool(row["active"]),
            last_login_utc=self.clock.now_utc(),
        )

    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole = UserRole.REVIEWER,
    ) -> User:
        """Register a new user with hashed password."""
        clean_user = username.strip().lower()
        if not clean_user:
            raise FastTrackError("Username cannot be empty.")

        existing = self.db.execute_query(
            "SELECT id FROM users WHERE lower(username) = ?",
            (clean_user,),
        )
        if existing:
            raise FastTrackError(f"Username '{clean_user}' already exists.")

        user_id = f"u-{uuid.uuid4().hex[:10]}"
        pwd_hash = hash_password(password)

        self.db.execute_non_query(
            """
            INSERT INTO users (id, username, password_hash, role, active)
            VALUES (?, ?, ?, ?, 1)
            """,
            (user_id, clean_user, pwd_hash, role.value),
        )

        logger.info("Created user %s with role %s", clean_user, role.value)
        return User(
            id=user_id,
            username=clean_user,
            password_hash=pwd_hash,
            role=role,
            active=True,
            last_login_utc=None,
        )

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Fetch user by primary key."""
        rows = self.db.execute_query(
            "SELECT id, username, password_hash, role, active, last_login_utc FROM users WHERE id = ?",
            (user_id,),
        )
        if not rows:
            return None
        row = rows[0]
        role_enum = UserRole(row["role"]) if row["role"] in UserRole.__members__.values() else UserRole.VIEWER
        return User(
            id=row["id"],
            username=row["username"],
            password_hash=row["password_hash"],
            role=role_enum,
            active=bool(row["active"]),
        )

    def list_users(self) -> List[User]:
        """List all registered system users."""
        rows = self.db.execute_query(
            "SELECT id, username, password_hash, role, active, last_login_utc FROM users ORDER BY username ASC"
        )
        users = []
        for row in rows:
            role_enum = UserRole(row["role"]) if row["role"] in UserRole.__members__.values() else UserRole.VIEWER
            users.append(
                User(
                    id=row["id"],
                    username=row["username"],
                    password_hash=row["password_hash"],
                    role=role_enum,
                    active=bool(row["active"]),
                )
            )
        return users
