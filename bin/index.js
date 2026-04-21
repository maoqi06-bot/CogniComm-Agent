#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');

const projectRoot = path.join(__dirname, '..');
const serverPath = path.join(projectRoot, 'dm_agent', 'rag', 'rag_mcp_server.py');

// Prefer the same Python environment that installed the project dependencies.
// Users can set PYTHON or PYTHON_EXECUTABLE in mcp_config.json env when needed.
function defaultPythonCommand() {
  if (process.env.CONDA_PREFIX) {
    return process.platform === 'win32'
      ? path.join(process.env.CONDA_PREFIX, 'python.exe')
      : path.join(process.env.CONDA_PREFIX, 'bin', 'python');
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

const pythonCmd = process.env.PYTHON || process.env.PYTHON_EXECUTABLE || defaultPythonCommand();

console.error('[Launcher] Starting RAG server...');
console.error(`[Launcher] Project root: ${projectRoot}`);
console.error(`[Launcher] Python: ${pythonCmd}`);

const pythonProcess = spawn(
  pythonCmd,
  [
    '-u',
    serverPath,
  ],
  {
    stdio: ['inherit', 'inherit', 'inherit'],
    cwd: projectRoot,
    env: {
      ...process.env,
      PYTHONPATH: projectRoot,
      PYTHONUNBUFFERED: '1',
    },
    shell: false,
  },
);

pythonProcess.on('error', (err) => {
  console.error(`[Launcher] Failed to start Python: ${err.message}`);
  console.error('Install Python dependencies with: python -m pip install -r requirements.txt');
  console.error('If you use a virtual environment, set PYTHON or PYTHON_EXECUTABLE in mcp_config.json.');
  process.exit(1);
});

pythonProcess.on('exit', (code) => {
  if (code !== 0 && code !== null) {
    console.error(`[Launcher] Python process exited with code ${code}`);
  }
});
