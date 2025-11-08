import { Pool } from 'pg';
import { connectionUrl } from '../shared/pg-db-conn';

const pool = new Pool({
  connectionString: connectionUrl()
});

export interface User {
  username: string;
  password: string;
}

export class UserRepo {
  constructor() {
    this.initTable().catch(err => console.error(err));
  }

  private async initTable() {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT NOT NULL
      );
    `);
  }

  async create(user: User): Promise<User> {
    const result = await pool.query<User>(
      `INSERT INTO users (username, password)
       VALUES ($1, $2)
       RETURNING *`,
      [user.username, user.password]
    );
    return result.rows[0];
  }

  async findAll(): Promise<User[]> {
    const result = await pool.query<User>('SELECT * FROM users ORDER BY username');
    return result.rows;
  }

  async findByUsername(username: string): Promise<User | null> {
    const result = await pool.query<User>(
      'SELECT * FROM users WHERE username = $1',
      [username]
    );
    return result.rows[0] || null;
  }

  async delete(username: string): Promise<boolean> {
    const result = await pool.query('DELETE FROM users WHERE username = $1', [username]);
    return parseInt((result.rowCount ?? 0)+'') > 0;
  }
}

export interface UserChatThread {
  id: string;
  username: string;
  title: string;
}

export class UserChatThreadRepo {
  constructor() {
    this.initTable().catch(err => console.error(err));
  }

  private async initTable() {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS user_chat_threads (
        id TEXT PRIMARY KEY,
        username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
        title TEXT NOT NULL
      );
    `);

    // Create an index for fast username lookups
    await pool.query(`
      CREATE INDEX IF NOT EXISTS idx_user_chat_threads_username
      ON user_chat_threads (username);
    `);
  }

  async createThread(thread: UserChatThread): Promise<UserChatThread> {
    const result = await pool.query<UserChatThread>(
      `INSERT INTO user_chat_threads (id, username, title)
       VALUES ($1, $2, $3)
       RETURNING *`,
      [thread.id, thread.username, thread.title]
    );
    return result.rows[0];
  }

  async findByUsername(username: string): Promise<UserChatThread[]> {
    const result = await pool.query<UserChatThread>(
      `SELECT * FROM user_chat_threads WHERE username = $1 ORDER BY id`,
      [username]
    );
    return result.rows;
  }


  async deleteThread(id: string): Promise<boolean> {
    const result = await pool.query('DELETE FROM user_chat_threads WHERE id = $1', [id]);
    return parseInt((result.rowCount ?? 0)+'') > 0;
  }
}

