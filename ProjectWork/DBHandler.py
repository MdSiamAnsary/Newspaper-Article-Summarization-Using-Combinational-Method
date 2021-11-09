import sqlite3


class Database:
    def __init__(self, db):
        self.conn = sqlite3.connect(db)
        self.cur = self.conn.cursor()

        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS summarytable (id INTEGER PRIMARY KEY, username text, url text, article text, summary text)")
        self.conn.commit()

    def fetch(self):
        self.cur.execute("SELECT * FROM summarytable")
        rows = self.cur.fetchall()
        return rows

    def insert(self, username, url, article, summary):
        self.cur.execute("INSERT INTO summarytable VALUES (NULL, ?, ?, ?, ?)",
                         (username, url, article, summary))
        self.conn.commit()

    def remove(self, id):
        self.cur.execute("DELETE FROM summarytable WHERE id=?", (id,))
        self.conn.commit()

    def update(self, username, url, article, summary):
        self.cur.execute("UPDATE summarytable SET username = ?, url = ?, article = ?, summary = ? WHERE id = ?",
                         (username, url, article, summary, id))
        self.conn.commit()

    def __del__(self):
        self.conn.close()


db = Database('summarizationDBFour.db')