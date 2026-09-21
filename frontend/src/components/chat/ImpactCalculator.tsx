import React, { useState, useEffect, useMemo } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Calculator, Info, RefreshCw } from "lucide-react"
import { InteractiveWidget, WidgetInput, WidgetOutput } from "./types"

interface ImpactCalculatorProps {
  widget: InteractiveWidget
  className?: string
}

export function ImpactCalculator({ widget, className }: ImpactCalculatorProps) {
  // Initialize state with default values or empty strings
  const [values, setValues] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {}
    widget.inputs.forEach(input => {
      if (input.default_value) {
        initial[input.name] = input.default_value
      }
    })
    return initial
  })

  // Handle input changes
  const handleInputChange = (name: string, value: string) => {
    setValues(prev => ({ ...prev, [name]: value }))
  }

  // Calculate results using useMemo to avoid useEffect setState cycles
  const { results, error } = useMemo(() => {
    const initialResults: Record<string, string> = {}
    
    // Check if we have values for all required inputs
    const hasAllInputs = widget.inputs.every(input => values[input.name] !== undefined && values[input.name] !== "")
    
    if (!hasAllInputs) {
      return { results: initialResults, error: null }
    }

    try {
      // Prepare variables for the formula
      const variables: Record<string, number> = {}
      let allValid = true

      widget.inputs.forEach(input => {
        const val = parseFloat(values[input.name] || "0")
        if (isNaN(val)) {
          allValid = false
        }
        variables[input.name] = val
      })

      if (!allValid) {
        return { results: initialResults, error: null }
      }

      // Create a safe evaluation context
      const argNames = Object.keys(variables)
      const argValues = Object.values(variables)
      
      // Wrap formula in a return statement if it doesn't have one
      let formulaBody = widget.formula
      if (!formulaBody.includes("return") && !formulaBody.includes(";")) {
        formulaBody = `return ${formulaBody};`
      } else if (!formulaBody.includes("return")) {
         if (!formulaBody.trim().startsWith("let") && !formulaBody.trim().startsWith("const") && !formulaBody.trim().startsWith("var") && !formulaBody.trim().startsWith("if")) {
             formulaBody = `return ${formulaBody};`
         }
      }

      const calculateFn = new Function(...argNames, formulaBody)
      const resultValue = calculateFn(...argValues)

      // Format outputs
      widget.outputs.forEach(output => {
        const formatted = output.format.replace("{value}", resultValue.toLocaleString(undefined, { maximumFractionDigits: 2 }))
        initialResults[output.label] = formatted
      })

      return { results: initialResults, error: null }
    } catch (err) {
      console.error("Calculation error:", err)
      return { results: initialResults, error: "Invalid calculation formula or inputs" }
    }
  }, [values, widget.formula, widget.inputs, widget.outputs])

  const handleReset = () => {
    const initial: Record<string, string> = {}
    widget.inputs.forEach(input => {
      if (input.default_value) {
        initial[input.name] = input.default_value
      }
    })
    setValues(initial)
  }

  return (
    <Card className={`w-full max-w-md border-primary/20 bg-primary/5 backdrop-blur-sm ${className}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <div className="p-2 rounded-lg bg-primary/10 text-primary">
            <Calculator className="w-5 h-5" />
          </div>
          <div>
            <CardTitle className="text-lg font-bold text-primary">{widget.title}</CardTitle>
            <CardDescription className="text-xs">{widget.description}</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-4">
          {widget.inputs.map((input) => (
            <div key={input.name} className="space-y-1.5">
              <label htmlFor={input.name} className="text-xs font-medium text-muted-foreground block">
                {input.label}
              </label>
              <Input
                id={input.name}
                type={input.type}
                placeholder={input.placeholder}
                value={values[input.name] || ""}
                onChange={(e) => handleInputChange(input.name, e.target.value)}
                className="bg-background/50 border-primary/10 focus:border-primary/30 transition-colors"
              />
            </div>
          ))}
        </div>

        {error && (
          <div className="p-2 rounded bg-red-500/10 text-red-500 text-xs">
            {error}
          </div>
        )}

        {Object.keys(results).length > 0 && (
          <div className="mt-4 pt-4 border-t border-primary/10 space-y-3 animate-in fade-in slide-in-from-top-2 duration-300">
            {widget.outputs.map((output, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-primary/10 border border-primary/20">
                <span className="text-sm font-medium text-muted-foreground">{output.label}</span>
                <span className="text-lg font-bold text-primary">{results[output.label]}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
      <CardFooter className="pt-0 pb-3 flex justify-between items-center text-xs text-muted-foreground">
        <div className="flex items-center gap-1">
            {widget.source_citation && (
                <>
                    <Info className="w-3 h-3" />
                    <span>Source: {widget.source_citation}</span>
                </>
            )}
        </div>
        <Button variant="ghost" size="sm" onClick={handleReset} className="h-6 px-2 text-xs hover:bg-primary/10">
            <RefreshCw className="w-3 h-3 mr-1" />
            Reset
        </Button>
      </CardFooter>
    </Card>
  )
}
