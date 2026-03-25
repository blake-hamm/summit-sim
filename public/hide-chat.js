// Hide the message composer
function hideMessageComposer() {
    const composer = document.getElementById('message-composer');
    if (composer) {
        composer.style.display = 'none';
    }
}

// Run immediately and keep trying
hideMessageComposer();
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', hideMessageComposer);
}
setInterval(hideMessageComposer, 500);
