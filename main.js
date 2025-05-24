const { app, BrowserWindow, ipcMain, dialog, screen, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);
const fs = require('fs');
const { PythonShell } = require('python-shell');

// Set the application name
app.setName('Reframe');

let mainWindow;
let currentProcess = null;
let processGroupId = null;
let isSettingUp = false;

function createWindow() {
  // Get the primary display's work area size
  const primaryDisplay = screen.getPrimaryDisplay();
  const { width, height } = primaryDisplay.workAreaSize;

  mainWindow = new BrowserWindow({
    width: width,
    height: height,
    minWidth: 1024, // Set minimum width as requested
    frame: true,
    autoHideMenuBar: true, // Keep menu bar hidden by default
    title: 'Reframe',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  // Maximize the window by default
  mainWindow.maximize();

  // Create the application menu (hidden by default but accessible with Alt key)
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Quit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'View',
      submenu: [
        {
          label: 'Toggle Developer Tools',
          accelerator: process.platform === 'darwin' ? 'Cmd+Option+I' : 'Ctrl+Shift+I',
          click: () => {
            mainWindow.webContents.toggleDevTools();
          }
        }
      ]
    }
  ];

  // Add standard Edit menu on macOS
  if (process.platform === 'darwin') {
    template.unshift({
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'delete' },
        { type: 'separator' },
        { role: 'selectAll' }
      ]
    });
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);

  mainWindow.loadFile('index.html');
  // DevTools will not open automatically
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// Handle file selection
ipcMain.handle('select-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Videos', extensions: ['mp4', 'mov', 'avi'] }
    ]
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

// Handle multiple file selection
ipcMain.handle('select-multiple-files', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Videos', extensions: ['mp4'] }
    ]
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths;
  }
  return null;
});

// Handle directory selection
ipcMain.handle('select-directory', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory']
  });
  
  if (!result.canceled && result.filePaths.length > 0) {
    return result.filePaths[0];
  }
  return null;
});

// Get video files in directory
ipcMain.handle('get-video-files-in-directory', async (event, dirPath, recursive = false) => {
  try {
    const videoFiles = [];
    const processDirectory = async (dir) => {
      const entries = await fs.promises.readdir(dir, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        
        if (entry.isDirectory() && recursive) {
          await processDirectory(fullPath);
        } else if (entry.isFile() && entry.name.toLowerCase().endsWith('.mp4')) {
          videoFiles.push(fullPath);
        }
      }
    };
    
    await processDirectory(dirPath);
    return videoFiles;
  } catch (error) {
    console.error('Error getting video files:', error);
    return null;
  }
});

// Function to check if setup is needed
async function checkSetup() {
    if (isSettingUp) {
        console.log('Setup already in progress...');
        return null; // Prevent multiple setup attempts
    }

    const appPath = app.getAppPath();
    const isDev = !app.isPackaged;
    const resourcesPath = isDev ? appPath : process.resourcesPath;
    const pythonPath = path.join(resourcesPath, 'python');
    const venvPath = path.join(pythonPath, 'python', 'venv');
    const setupCompletePath = path.join(venvPath, '.setup_complete');

    console.log('Environment check:');
    console.log('- App path:', appPath);
    console.log('- Resources path:', resourcesPath);
    console.log('- Python path:', pythonPath);
    console.log('- Venv path:', venvPath);
    console.log('- Setup complete path:', setupCompletePath);
    console.log('- Is dev mode:', isDev);
    console.log('- Current working directory:', process.cwd());

    // If setup is already complete, return the Python path
    if (fs.existsSync(setupCompletePath)) {
        console.log('Python environment already set up');
        const pythonExecutable = path.join(venvPath, process.platform === 'win32' ? 'Scripts' : 'bin', 'python');
        console.log('Using Python executable:', pythonExecutable);
        return pythonExecutable;
    }

    console.log('Python environment needs setup');
    isSettingUp = true;

    try {
        // Show setup dialog
        const { response } = await dialog.showMessageBox(mainWindow, {
            type: 'info',
            title: 'First Run Setup',
            message: 'Reframe needs to set up its Python environment. This may take a few minutes.',
            detail: 'The setup process will:\n1. Create a Python virtual environment\n2. Install required packages\n3. Download AI models\n\nProgress will be shown in the debug panel.',
            buttons: ['OK', 'Cancel'],
            defaultId: 0
        });

        if (response === 1) { // Cancel
            console.log('Setup cancelled by user');
            app.quit();
            return null;
        }

        // Show debug panel and enable debug mode
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('show-debug-panel');
            mainWindow.webContents.send('python-output', 'Starting Python environment setup...\n');
        }

        // Run setup script using system Python
        const setupScript = path.join(pythonPath, 'setup.py');
        console.log('Running setup script:', setupScript);
        
        // Use system Python instead of Electron executable
        const pythonCommand = process.platform === 'win32' ? 'python' : 'python3';
        console.log('Using Python command:', pythonCommand);
        
        const setupProcess = spawn(pythonCommand, [setupScript], {
            stdio: 'pipe',
            env: { 
                ...process.env, 
                PYTHONUNBUFFERED: '1',
                PYTHONPATH: pythonPath
            },
            cwd: pythonPath // Set working directory to python directory
        });

        let setupOutput = '';
        setupProcess.stdout.on('data', (data) => {
            const message = data.toString();
            setupOutput += message;
            console.log('Setup output:', message);
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('python-output', message);
            }
        });

        setupProcess.stderr.on('data', (data) => {
            const message = data.toString();
            setupOutput += message;
            console.error('Setup error:', message);
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('python-output', `Error: ${message}`);
            }
        });

        return new Promise((resolve, reject) => {
            setupProcess.on('close', (code) => {
                isSettingUp = false;
                console.log('Setup process closed with code:', code);
                if (code === 0) {
                    console.log('Setup completed successfully');
                    if (mainWindow && !mainWindow.isDestroyed()) {
                        mainWindow.webContents.send('python-output', '\nSetup completed successfully!\n');
                    }
                    const pythonExecutable = path.join(venvPath, process.platform === 'win32' ? 'Scripts' : 'bin', 'python');
                    console.log('Using Python executable:', pythonExecutable);
                    resolve(pythonExecutable);
                } else {
                    console.error('Setup failed with code:', code);
                    console.error('Setup output:', setupOutput);
                    if (mainWindow && !mainWindow.isDestroyed()) {
                        mainWindow.webContents.send('python-output', '\nSetup failed!\n');
                        dialog.showErrorBox(
                            'Setup Failed',
                            'Failed to set up Python environment. Please try reinstalling the application.\n\n' + setupOutput
                        );
                    }
                    reject(new Error('Setup failed'));
                }
            });

            setupProcess.on('error', (err) => {
                isSettingUp = false;
                console.error('Setup process error:', err);
                if (mainWindow && !mainWindow.isDestroyed()) {
                    mainWindow.webContents.send('python-output', `\nSetup error: ${err.message}\n`);
                    dialog.showErrorBox(
                        'Setup Error',
                        'An error occurred while setting up the Python environment: ' + err.message
                    );
                }
                reject(err);
            });
        });
    } catch (error) {
        isSettingUp = false;
        console.error('Setup error:', error);
        throw error;
    }
}

// Handle Python script execution
ipcMain.handle('run-python-script', async (event, videoPath, options = {}) => {
    try {
        const pythonPath = await checkSetup();
        if (!pythonPath) {
            throw new Error('Python environment not available');
        }

        const appPath = app.getAppPath();
        const isDev = !app.isPackaged;
        const resourcesPath = isDev ? appPath : process.resourcesPath;
        const pythonDir = path.join(resourcesPath, 'python');

        // Generate output path
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const pathParts = videoPath.split('/').pop().split('.');
        const extension = pathParts.pop();
        const defaultFilename = pathParts.join('.');
        const outputFilename = options.output_filename || `${defaultFilename}_${timestamp}`;
        const outputPath = path.join(options.output_dir, `${outputFilename}.${extension}`);

        // Build command line arguments array
        const args = [
            '--input', videoPath,
            '--output', outputPath
        ];

        // Add all optional parameters if they are provided
        if (options.target_ratio !== undefined) args.push('--target_ratio', options.target_ratio.toString());
        if (options.max_workers !== undefined) args.push('--max_workers', options.max_workers.toString());
        if (options.detector !== undefined) args.push('--detector', options.detector);
        if (options.skip_frames !== undefined) args.push('--skip_frames', options.skip_frames.toString());
        if (options.conf_threshold !== undefined) args.push('--conf_threshold', options.conf_threshold.toString());
        if (options.model_size !== undefined) args.push('--model_size', options.model_size);
        if (options.object_classes !== undefined) args.push('--object_classes', ...options.object_classes.map(x => x.toString()));
        if (options.track_count !== undefined) args.push('--track_count', options.track_count.toString());
        if (options.padding_ratio !== undefined) args.push('--padding_ratio', options.padding_ratio.toString());
        if (options.size_weight !== undefined) args.push('--size_weight', options.size_weight.toString());
        if (options.center_weight !== undefined) args.push('--center_weight', options.center_weight.toString());
        if (options.motion_weight !== undefined) args.push('--motion_weight', options.motion_weight.toString());
        if (options.history_weight !== undefined) args.push('--history_weight', options.history_weight.toString());
        if (options.saliency_weight !== undefined) args.push('--saliency_weight', options.saliency_weight.toString());
        if (options.face_detection) args.push('--face_detection');
        if (options.weighted_center) args.push('--weighted_center');
        if (options.blend_saliency) args.push('--blend_saliency');
        if (options.apply_smoothing) args.push('--apply_smoothing');
        if (options.smoothing_window !== undefined) args.push('--smoothing_window', options.smoothing_window.toString());
        if (options.position_inertia !== undefined) args.push('--position_inertia', options.position_inertia.toString());
        if (options.size_inertia !== undefined) args.push('--size_inertia', options.size_inertia.toString());
        if (options.debug) args.push('--debug');

        console.log('Starting Python process:');
        console.log('- Python executable:', pythonPath);
        console.log('- Working directory:', pythonDir);
        console.log('- Arguments:', args.join(' '));

        // Use the Python from our virtual environment
        currentProcess = spawn(pythonPath, ['-u', 'main.py', ...args], {
            cwd: pythonDir, // Set working directory to python directory
            detached: true,
            stdio: ['ignore', 'pipe', 'pipe'],
            env: { 
                ...process.env,
                PYTHONPATH: pythonDir
            }
        });

        // Store the process group ID (on Unix-like systems, this is the same as the PID)
        processGroupId = currentProcess.pid;
        console.log('Process started with PID:', processGroupId);

        // Unref the parent process to allow it to exit independently
        currentProcess.unref();

        // Handle process output
        currentProcess.stdout.on('data', (data) => {
            const message = data.toString().trim();
            if (message) {
                console.log('Python stdout:', message);
                event.sender.send('python-output', message);
            }
        });

        currentProcess.stderr.on('data', (data) => {
            const message = data.toString().trim();
            if (message) {
                console.error('Python stderr:', message);
                event.sender.send('python-output', `Error: ${message}`);
            }
        });

        // Handle process completion
        currentProcess.on('close', (code) => {
            console.log('Process closed with code:', code);
            currentProcess = null;
            if (code === 0) {
                event.sender.send('python-script-complete', { success: true, outputPath });
            } else {
                event.sender.send('python-script-complete', { 
                    success: false, 
                    error: `Process exited with code ${code}`,
                    code: 'PROCESS_ERROR'
                });
            }
        });

        currentProcess.on('error', (err) => {
            console.error('Process error:', err);
            currentProcess = null;
            event.sender.send('python-script-complete', { 
                success: false, 
                error: err.message || err.toString(),
                stack: err.stack || '',
                code: err.code || 'UNKNOWN_ERROR'
            });
        });
    } catch (error) {
        console.error('Error running Python script:', error);
        if (error.message === 'Setup failed') {
            app.quit();
        }
        event.sender.send('python-script-complete', { 
            success: false, 
            error: error.message || 'Failed to run Python script',
            stack: error.stack || '',
            code: error.code || 'SCRIPT_ERROR'
        });
    }
});

// Handle cancellation
ipcMain.handle('cancel-processing', async () => {
    console.log('Cancel requested, process group ID:', processGroupId);
    
    if (!processGroupId) {
        console.log('No process to cancel');
        return { success: false, error: 'No process running' };
    }

    try {
        if (process.platform === 'win32') {
            // Windows
            console.log('Using taskkill on Windows');
            await execPromise(`taskkill /F /T /PID ${processGroupId}`);
        } else {
            // Unix-like (macOS/Linux)
            console.log('Using kill on Unix-like system');
            
            // First try SIGTERM
            try {
                process.kill(-processGroupId, 'SIGTERM');
                console.log('Sent SIGTERM to process group');
                
                // Wait a bit to see if processes terminate gracefully
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                // Check if process is still running
                try {
                    process.kill(-processGroupId, 0);
                    console.log('Process still running, sending SIGKILL');
                    // If we get here, process is still running, so send SIGKILL
                    process.kill(-processGroupId, 'SIGKILL');
                } catch (e) {
                    console.log('Process terminated after SIGTERM');
                }
            } catch (e) {
                console.log('Error sending SIGTERM, trying SIGKILL directly');
                process.kill(-processGroupId, 'SIGKILL');
            }
        }

        console.log('Process kill command sent');
        currentProcess = null;
        processGroupId = null;
        return { success: true };
    } catch (error) {
        console.error('Error during cancellation:', error);
        // Even if we get an error, try to clean up
        currentProcess = null;
        processGroupId = null;
        return { 
            success: false, 
            error: error.message || 'Failed to cancel processing' 
        };
    }
}); 