import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

export default function RoleSelection() {
    return (
        <div className="grid grid-cols-2 gap-6 w-full max-w-4xl">
            <Card className="flex flex-col">
                <CardHeader className="text-center">
                    <div className="text-5xl mb-3">🎓</div>
                    <CardTitle className="text-xl">Instructor</CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col flex-grow space-y-4">
                    <p className="text-sm text-muted-foreground">
                        Create and review scenarios with full details visible.
                    </p>
                    <ul className="list-disc pl-5 space-y-2 text-sm flex-grow">
                        <li>View hidden medical information</li>
                        <li>Provide feedback to refine scenarios</li>
                        <li>Share scenarios with students when ready</li>
                    </ul>
                    <Button 
                        onClick={() => submitElement({ role: "instructor" })}
                        className="w-full mt-4"
                        size="lg"
                        style={{ backgroundColor: '#dc2626', color: 'white' }}
                    >
                        Create Scenario
                    </Button>
                </CardContent>
            </Card>

            <Card className="flex flex-col">
                <CardHeader className="text-center">
                    <div className="text-5xl mb-3">👤</div>
                    <CardTitle className="text-xl">Student</CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col flex-grow space-y-4">
                    <p className="text-sm text-muted-foreground">
                        Self-directed practice with immediate simulation start.
                    </p>
                    <ul className="list-disc pl-5 space-y-2 text-sm flex-grow">
                        <li>No hidden information shown</li>
                        <li>Discover medical details through assessment</li>
                        <li>Configure and play scenarios instantly</li>
                    </ul>
                    <Button 
                        onClick={() => submitElement({ role: "student" })}
                        className="w-full mt-4"
                        size="lg"
                        style={{ backgroundColor: '#dc2626', color: 'white' }}
                    >
                        Start Practice
                    </Button>
                </CardContent>
            </Card>
        </div>
    );
}
