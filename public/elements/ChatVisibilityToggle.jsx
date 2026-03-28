import { useEffect } from 'react';

export default function ChatVisibilityToggle() {
    const visible = props?.visible ?? false;
    
    useEffect(() => {
        const composer = document.getElementById('message-composer');
        if (composer) {
            composer.style.display = visible ? 'flex' : 'none';
        }
        
        // Cleanup: hide chat when component unmounts
        return () => {
            if (composer) {
                composer.style.display = 'none';
            }
        };
    }, [visible]);
    
    // No visible UI - just controls chat visibility
    return null;
}
