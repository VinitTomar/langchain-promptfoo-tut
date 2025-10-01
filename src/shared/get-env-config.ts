import dotenv from 'dotenv';
import path from 'path';

dotenv.config({
  path: path.join(__dirname, '../../.env')
});

export default {
  GOOGLE_API_KEY: <string>process.env.GOOGLE_API_KEY,
  TAVILY_API_KEY: <string>process.env.TAVILY_API_KEY,
}