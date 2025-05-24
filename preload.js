const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectFile: () => ipcRenderer.invoke('select-file'),
  selectDirectory: () => ipcRenderer.invoke('select-directory'),
  selectMultipleFiles: () => ipcRenderer.invoke('select-multiple-files'),
  getVideoFilesInDirectory: (dirPath, recursive) => ipcRenderer.invoke('get-video-files-in-directory', dirPath, recursive),
  runPythonScript: (videoPath, options) => ipcRenderer.invoke('run-python-script', videoPath, options),
  cancelProcessing: () => ipcRenderer.invoke('cancel-processing'),
  onPythonOutput: (callback) => {
    ipcRenderer.on('python-output', callback);
    return () => ipcRenderer.removeListener('python-output', callback);
  },
  onPythonScriptComplete: (callback) => {
    ipcRenderer.on('python-script-complete', callback);
    return () => ipcRenderer.removeListener('python-script-complete', callback);
  },
  removePythonOutputListener: (callback) => ipcRenderer.removeListener('python-output', callback),
  removePythonScriptCompleteListener: (callback) => ipcRenderer.removeListener('python-script-complete', callback),
  onShowDebugPanel: (callback) => {
    ipcRenderer.on('show-debug-panel', callback);
  }
}); 