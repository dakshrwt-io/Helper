module.exports = {
  apps: [
    {
      name: 'ai-agent',
      script: 'run_server.py',
      interpreter: 'C:\\Projects\\.venv\\Scripts\\python.exe',
      cwd: 'C:\\Projects\\Helper',
      env: { PYTHONPATH: '.' },
      watch: false,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 3000,
      max_memory_restart: '500M',
      error_file: 'pm2_logs/agent.err.log',
      out_file: 'pm2_logs/agent.out.log',
      time: true,
    },
  ],
};
