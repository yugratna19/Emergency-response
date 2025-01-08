document.addEventListener('DOMContentLoaded', function() {
    // Handle video upload
    const videoUpload = document.getElementById('video-upload');
    const detectionView = document.getElementById('detection-view');
    const detectionVideo = document.getElementById('detection-video');
    const resultsContent = document.getElementById('results-content');

    videoUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const videoUrl = URL.createObjectURL(file);
            detectionVideo.src = videoUrl;
            detectionView.style.display = 'block';
            document.getElementById('detection-options').style.display = 'none';
            
            // Simulate detection results
            showDetectionResults();
        }
    });
});

function startRealtimeDetection() {
    const detectionView = document.getElementById('detection-view');
    detectionView.style.display = 'block';
    document.getElementById('detection-options').style.display = 'none';
    
    document.getElementById('results-content').innerHTML = `
        <div class="detection-status">
            <div class="status-item">
                <h4>Connection Status</h4>
                <p class="status-active">Connected to CCTV feed</p>
            </div>
            <div class="status-item">
                <h4>Processing</h4>
                <p>Real-time analysis active</p>
            </div>
            <div class="status-item">
                <h4>Live Events</h4>
                <ul class="events-list">
                    <li class="event-item">Monitoring for suspicious activity...</li>
                </ul>
            </div>
        </div>
    `;
}

function showDetectionResults() {
    // Simulate detection results with more detailed content
    document.getElementById('results-content').innerHTML = `
        <div class="detection-status">
            <div class="status-item">
                <h4>Processing Status</h4>
                <p>Analyzing video feed...</p>
            </div>
            <div class="status-item">
                <h4>Current Frame</h4>
                <p>Frame: 127/500</p>
            </div>
            <div class="status-item">
                <h4>Detected Events</h4>
                <ul class="events-list">
                    <li>
                        <span class="timestamp">00:15</span>
                        <span class="event-type">Suspicious Activity</span>
                    </li>
                    <li>
                        <span class="timestamp">00:23</span>
                        <span class="event-type">Movement Detected</span>
                    </li>
                </ul>
            </div>
        </div>
    `;
} 