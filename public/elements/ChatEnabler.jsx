import React, { useEffect } from 'react';

export default function ChatEnabler() {
    useEffect(() => {
        // Switch to player mode to show chat input
        document.body.classList.remove('author-mode');
        document.body.classList.add('player-mode');
    }, []);

    return null;
}
