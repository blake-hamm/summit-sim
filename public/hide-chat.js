// Detect mode from URL and apply appropriate CSS class
function detectAndApplyMode() {
    const urlParams = new URLSearchParams(window.location.search);
    const scenarioId = urlParams.get('scenario_id');

    if (scenarioId) {
        // Player mode - show chat input (don't hide it)
        document.body.classList.add('player-mode');
        document.body.classList.remove('author-mode');
    } else {
        // Author mode - hide chat input
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
