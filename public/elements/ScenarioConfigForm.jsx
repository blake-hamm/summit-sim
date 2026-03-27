import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"

export default function ScenarioConfigForm() {
    const fields = props?.fields || [];

    const [values, setValues] = useState(() => {
        const init = {};
        fields.forEach((f) => {
            if (f.value) {
                init[f.id] = f.value;
            } else if (f.type === 'select' && f.options && f.options.length > 0) {
                init[f.id] = f.options[1];
            } else {
                init[f.id] = '';
            }
        });
        return init;
    });

    const allValid = fields.every((f) => {
        if (!f.required) return true;
        const val = values[f.id];
        return val !== undefined && val !== '';
    });

    const handleChange = (id, val) => {
        setValues((v) => ({ ...v, [id]: val }));
    };

    const renderField = (field) => {
        const value = values[field.id];
        switch (field.type) {
            case 'select':
                return (
                    <Select
                        value={value}
                        onValueChange={(val) => handleChange(field.id, val)}
                    >
                        <SelectTrigger className="w-full">
                            <SelectValue placeholder="Select..." />
                        </SelectTrigger>
                        <SelectContent>
                            {field.options.map((opt) => (
                                <SelectItem key={opt} value={opt}>
                                    {opt}
                                </SelectItem>
                            ))}
                        </SelectContent>
                    </Select>
                );
            default:
                return (
                    <input
                        id={field.id}
                        type="text"
                        value={value}
                        onChange={(e) => handleChange(field.id, e.target.value)}
                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                    />
                );
        }
    };

    if (!fields || fields.length === 0) {
        return (
            <div className="p-4 text-destructive">
                Error: No fields provided. Check console for details.
            </div>
        );
    }

    return (
        <Card className="w-full max-w-md mt-2">
            <CardHeader>
                <CardTitle className="text-lg">Configure Scenario</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                {fields.map((field) => (
                    <div key={field.id} className="space-y-2">
                        <Label htmlFor={field.id}>
                            {field.label}
                            {field.required && (
                                <span className="text-destructive ml-1">*</span>
                            )}
                        </Label>
                        {renderField(field)}
                    </div>
                ))}
            </CardContent>
            <CardFooter className="flex justify-end gap-2">
                <Button
                    variant="outline"
                    onClick={() => cancelElement()}
                >
                    Cancel
                </Button>
                <Button
                    onClick={() => submitElement(values)}
                    disabled={!allValid}
                >
                    Submit
                </Button>
            </CardFooter>
        </Card>
    );
}
