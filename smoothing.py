import numpy as np
import os

class CropWindowSmoother:
    """Class for applying temporal smoothing to crop windows."""
    
    def __init__(self, window_size=30, position_inertia=0.8, size_inertia=0.9):
        """
        Initialize the crop window smoother.
        
        Args:
            window_size: Size of the smoothing window (in frames)
            position_inertia: Inertia factor for position changes (0-1)
            size_inertia: Inertia factor for size changes (0-1)
        """
        self.window_size = window_size
        self.position_inertia = position_inertia
        self.size_inertia = size_inertia
    
    def smooth(self, crop_windows):
        """
        Apply temporal smoothing to crop windows.
        
        Args:
            crop_windows: List of crop windows, each as [x, y, width, height]
            
        Returns:
            List of smoothed crop windows
        """
        if not crop_windows:
            return []
        
        # Convert to numpy array for easier manipulation
        windows = np.array(crop_windows)
        
        # Apply moving average smoothing
        smoothed_windows = self._moving_average(windows)
        
        # Apply inertia-based smoothing
        smoothed_windows = self._apply_inertia(smoothed_windows)
        
        # Convert back to list of lists
        return smoothed_windows.astype(int).tolist()
    
    def _moving_average(self, windows):
        """Apply moving average smoothing to crop windows."""
        n_frames = len(windows)
        smoothed = np.copy(windows).astype(float)
        
        half_window = self.window_size // 2
        
        for i in range(n_frames):
            # Calculate window bounds
            start = max(0, i - half_window)
            end = min(n_frames, i + half_window + 1)
            
            # Calculate weights (higher for frames closer to current frame)
            weights = np.exp(-0.5 * ((np.arange(start, end) - i) / (half_window / 2)) ** 2)
            weights = weights / np.sum(weights)
            
            # Apply weighted average
            smoothed[i] = np.sum(windows[start:end] * weights[:, np.newaxis], axis=0)
        
        return smoothed
    
    def _apply_inertia(self, windows):
        """Apply inertia-based smoothing to crop windows."""
        n_frames = len(windows)
        smoothed = np.copy(windows)
        
        for i in range(1, n_frames):
            # Split into position (x, y) and size (width, height)
            prev_pos = smoothed[i-1, :2]
            prev_size = smoothed[i-1, 2:]
            
            curr_pos = windows[i, :2]
            curr_size = windows[i, 2:]
            
            # Apply inertia
            new_pos = prev_pos * self.position_inertia + curr_pos * (1 - self.position_inertia)
            new_size = prev_size * self.size_inertia + curr_size * (1 - self.size_inertia)
            
            # Update smoothed window
            smoothed[i, :2] = new_pos
            smoothed[i, 2:] = new_size
        
        return smoothed 