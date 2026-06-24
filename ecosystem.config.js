module.exports = {
  apps: [
    {
      name: 'ai-agent',
      script: 'run_server.py',
      interpreter: './.venv/Scripts/python.exe',
      cwd: 'C:\\Personal ai agent',
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
