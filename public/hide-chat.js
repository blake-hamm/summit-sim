// Detect mode from URL and apply appropriate CSS class
function detectAndApplyMode() {
    const urlParams = new URLSearchParams(window.location.search);
    const scenarioId = urlParams.get('scenario_id');

    if (scenarioId) {
        // Player mode - show chat input (don't hide it)
        setMode('player');
    } else {
        // Author mode - hide chat input
        setMode('author');
    }
}

function setMode(mode) {
    if (mode === 'player') {
        document.body.classList.add('player-mode');
        document.body.classList.remove('author-mode');
    } else {
        document.body.classList.add('author-mode');
        document.body.classList.remove('player-mode');
    }
}

// Apply immediately
detectAndApplyMode();

// Also apply when DOM is fully loaded (in case of dynamic loading)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', detectAndApplyMode);
}

// Listen for mode changes from backend via custom events
window.addEventListener('summit-mode-change', (e) => {
    if (e.detail && e.detail.mode) {
        setMode(e.detail.mode);
    }
});

// Track if we've already switched to player mode (to avoid redundant operations)
let isPlayerMode = document.body.classList.contains('player-mode');

// Zero-width non-joiner character used as signal
const SIMULATION_READY_MARKER = '\u200C';

// Watch for simulation ready signal in messages
function watchForSimulationReady() {
    const observer = new MutationObserver((mutations) => {
        if (isPlayerMode) return; // Already in player mode, no need to check
        
        mutations.forEach((mutation) => {
            mutation.addedNodes.forEach((node) => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    // Check if the added node or its children contain the marker
                    const text = node.textContent || '';
                    if (text.includes(SIMULATION_READY_MARKER)) {
                        setMode('player');
                        isPlayerMode = true;
                    }
                }
            });
        });
    });

    // Start observing the document body for changes
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
}

// Start watching for simulation ready signal
watchForSimulationReady();
