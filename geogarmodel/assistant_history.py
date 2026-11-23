"""Local storage for assistant chats and bounded conversation context."""
import json
import sqlite3
from contextlib import closing, contextmanager, nullcontext
from dataclasses import asdict, dataclass

from .manual_assistant import Passage, terms


@dataclass(frozen=True)
class Message:
    role: str
    text: str
    mode: str
    sources: tuple = ()
    excerpt: bool = False


class ChatStore:
    def __init__(self, path):
        self.path = path
        self.memory = sqlite3.connect(':memory:') if path is None else None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY,
                    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, id);
            """)

    @contextmanager
    def connect(self):
        connection = self.memory if self.memory is not None else sqlite3.connect(self.path)
        manager = nullcontext(connection) if self.memory is not None else closing(connection)
        with manager as db, db:
            db.execute("PRAGMA foreign_keys = ON")
            yield db

    def chats(self):
        with self.connect() as db:
            return db.execute("SELECT id, title FROM chats ORDER BY id DESC").fetchall()

    def create(self):
        with self.connect() as db:
            return db.execute("INSERT INTO chats(title) VALUES (?)", ("Новый чат",)).lastrowid

    def rename(self, chat_id, title):
        with self.connect() as db:
            db.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))

    def delete(self, chat_id):
        with self.connect() as db:
            db.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def append(self, chat_id, message):
        with self.connect() as db:
            db.execute("INSERT INTO messages(chat_id, data) VALUES (?, ?)",
                       (chat_id, json.dumps(asdict(message), ensure_ascii=False)))

    def messages(self, chat_id):
        with self.connect() as db:
            rows = db.execute("SELECT data FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,))
            messages = []
            for (row,) in rows:
                data = json.loads(row)
                data["sources"] = tuple(Passage(**source) for source in data["sources"])
                messages.append(Message(**data))
            return messages


def conversation_context(messages, question, mode):
    """Include recent questions and relevant older context from this chat."""
    turns = []
    for message in messages:
        if message.role == "user":
            turns.append({"question": message.text})
        elif (turns and message.role == "assistant" and message.mode == mode
              and message.text and message.sources and not message.excerpt):
            turns[-1]["answer"] = message.text

    query = terms(question)
    recent = turns[-2:]
    older = turns[:-2]
    older.sort(key=lambda turn: len(query & terms(turn["question"])), reverse=True)
    related = [turn for turn in older if query & terms(turn["question"])][:2]
    return [
        {key: text[:260] for key, text in turn.items()}
        for turn in [*related, *recent]
    ]
