const dbUsername = 'dev_user';
const dbPassword = 'dev_password';
const dbHost = 'pg-vector-store';
const dbPort = '5432';
const dbName = 'langchain_db';


function connectionUrl() {
  return `postgresql://${dbUsername}:${dbPassword}@${dbHost}:${dbPort}/${dbName}`
}

export {
  connectionUrl
}