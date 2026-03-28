import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"

export default function RevisionFeedbackForm() {
    const [feedback, setFeedback] = useState('');
    
    const handleSubmit = () => {
        if (feedback.trim()) {
            submitElement({ output: feedback.trim() });
        }
    };
    
    const handleCancel = () => {
        cancelElement();
    };
    
    return (
        <Card className="w-full max-w-lg mt-2">
            <CardHeader>
                <CardTitle className="text-lg">Revise Scenario</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                <p className="text-sm text-muted-foreground">
                    What would you like me to change? Be specific about what you'd like different.
                </p>
                <Textarea
                    placeholder="e.g., 'Make the injuries more severe', 'Change the setting to winter', 'Add more complex symptoms'..."
                    value={feedback}
                    onChange={(e) => setFeedback(e.target.value)}
                    className="min-h-[100px]"
                />
            </CardContent>
            <CardFooter className="flex justify-end gap-2">
                <Button
                    variant="outline"
                    onClick={handleCancel}
                >
                    Cancel
                </Button>
                <Button
                    onClick={handleSubmit}
                    disabled={!feedback.trim()}
                >
                    Submit Feedback
                </Button>
            </CardFooter>
        </Card>
    );
}
