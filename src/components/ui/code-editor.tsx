"use client";

import { useState } from "react";
import { Button } from "./button";

interface CodeEditorProps {
  initialCode: string;
  language?: string;
  showOutput?: boolean;
  expectedOutput?: string;
  isExercise?: boolean;
  solution?: string;
}

export function CodeEditor({
  initialCode,
  language = "python",
  showOutput = true,
  expectedOutput,
  isExercise = false,
  solution,
}: CodeEditorProps) {
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState<string>("");
  const [isRunning, setIsRunning] = useState(false);
  const [showSolution, setShowSolution] = useState(false);

  const mockInterpreter = (code: string): string => {
    // Simple mock interpreter for basic Python-like code
    const lines = code.split('\n').map(line => line.trim()).filter(line => line && !line.startsWith('#'));
    const variables: Record<string, unknown> = {};
    const functions: Record<string, { params: string[], body: string[] }> = {};
    const outputLines: string[] = [];
    let indentLevel = 0;
    let inFunction = false;
    let currentFunction = '';
    let skipLines = 0;

    const evaluateExpression = (expr: string): unknown => {
      // Very basic expression evaluator
      expr = expr.trim();

      // String literals
      if ((expr.startsWith('"') && expr.endsWith('"')) || (expr.startsWith("'") && expr.endsWith("'"))) {
        return expr.slice(1, -1);
      }

      // Numbers
      if (!isNaN(Number(expr))) {
        return Number(expr);
      }

      // Variables
      if (variables[expr] !== undefined) {
        return variables[expr];
      }

      // Function calls like range()
      if (expr.startsWith('range(') && expr.endsWith(')')) {
        const args = expr.slice(6, -1).split(',').map(a => evaluateExpression(a.trim()));
        if (args.length === 1) {
          return Array.from({length: args[0] as number}, (_, i) => i);
        } else if (args.length === 2) {
          return Array.from({length: (args[1] as number) - (args[0] as number)}, (_, i) => i + (args[0] as number));
        }
      }

      // List literals
      if (expr.startsWith('[') && expr.endsWith(']')) {
        const items = expr.slice(1, -1).split(',').map(item => evaluateExpression(item.trim()));
        return items;
      }

      // Simple operations
      const ops = ['+', '-', '*', '/', '**', '//', '%'];
      for (const op of ops) {
        if (expr.includes(op)) {
          const parts = expr.split(op).map(p => p.trim());
          if (parts.length === 2) {
            const left = evaluateExpression(parts[0]) as number;
            const right = evaluateExpression(parts[1]) as number;
            switch (op) {
              case '+': return left + right;
              case '-': return left - right;
              case '*': return left * right;
              case '/': return left / right;
              case '**': return Math.pow(left, right);
              case '//': return Math.floor(left / right);
              case '%': return left % right;
            }
          }
        }
      }

      // Comparisons
      const comps = ['>', '<', '>=', '<=', '==', '!='];
      for (const comp of comps) {
        if (expr.includes(comp)) {
          const parts = expr.split(comp).map(p => p.trim());
          if (parts.length === 2) {
            const left = evaluateExpression(parts[0]) as number;
            const right = evaluateExpression(parts[1]) as number;
            switch (comp) {
              case '>': return left > right;
              case '<': return left < right;
              case '>=': return left >= right;
              case '<=': return left <= right;
              case '==': return left == right;
              case '!=': return left != right;
            }
          }
        }
      }

      return expr;
    };

    for (let i = 0; i < lines.length; i++) {
      if (skipLines > 0) {
        skipLines--;
        continue;
      }

      const line = lines[i];
      const indent = line.length - line.trimStart().length;

      if (line.startsWith('def ')) {
        // Function definition
        const match = line.match(/def (\w+)\((.*?)\):/);
        if (match) {
          const [, funcName, params] = match;
          functions[funcName] = {
            params: params.split(',').map(p => p.trim()).filter(p => p),
            body: []
          };
          inFunction = true;
          currentFunction = funcName;
          indentLevel = indent + 4;
        }
      } else if (inFunction) {
        if (indent < indentLevel && line.trim()) {
          inFunction = false;
          currentFunction = '';
        } else if (line.trim()) {
          functions[currentFunction].body.push(line);
        }
      } else if (line.startsWith('if ')) {
        const condition = line.match(/if (.*?):/);
        if (condition) {
          const result = evaluateExpression(condition[1]);
          if (!result) {
            // Skip until else/elif or end of block
            let j = i + 1;
            while (j < lines.length) {
              const nextLine = lines[j];
              const nextIndent = nextLine.length - nextLine.trimStart().length;
              if (nextIndent <= indent && nextLine.trim()) break;
              if (nextLine.startsWith('else:') || nextLine.startsWith('elif ')) {
                // Handle else/elif - for simplicity, skip to end
                break;
              }
              j++;
            }
            skipLines = j - i - 1;
          }
        }
      } else if (line.startsWith('while ')) {
        const match = line.match(/while (.+?):/);
        if (match) {
          const condition = match[1];
          const bodyLines = [];
          let j = i + 1;
          while (j < lines.length) {
            const nextLine = lines[j];
            const nextIndent = nextLine.length - nextLine.trimStart().length;
            if (nextIndent <= indent && nextLine.trim()) break;
            bodyLines.push(nextLine);
            j++;
          }

          let iterations = 0;
          while (evaluateExpression(condition) && iterations < 100) { // Prevent infinite loops
            for (const bodyLine of bodyLines) {
              const trimmed = bodyLine.trim();
              if (trimmed.startsWith('print(') && trimmed.endsWith(')')) {
                const content = trimmed.slice(6, -1);
                const value = evaluateExpression(content);
                outputLines.push(String(value));
              } else if (trimmed.includes('=')) {
                const [vName, vValue] = trimmed.split('=').map(s => s.trim());
                variables[vName] = evaluateExpression(vValue);
              }
            }
            iterations++;
          }
          skipLines = j - i - 1;
        }
      } else if (line.startsWith('for ')) {
        const match = line.match(/for (\w+) in (.+?):/);
        if (match) {
          const [, varName, iterableExpr] = match;
          const iterable = evaluateExpression(iterableExpr);
          if (Array.isArray(iterable) || typeof iterable === 'string') {
            // For loop over array or string
            const bodyLines = [];
            let j = i + 1;
            while (j < lines.length) {
              const nextLine = lines[j];
              const nextIndent = nextLine.length - nextLine.trimStart().length;
              if (nextIndent <= indent && nextLine.trim()) break;
              bodyLines.push(nextLine);
              j++;
            }

            for (const item of iterable) {
              variables[varName] = item;
              // Execute body (simplified - only handles print for now)
              for (const bodyLine of bodyLines) {
                const trimmed = bodyLine.trim();
                if (trimmed.startsWith('print(') && trimmed.endsWith(')')) {
                  const content = trimmed.slice(6, -1);
                  const value = evaluateExpression(content);
                  outputLines.push(String(value));
                } else if (trimmed.includes('=')) {
                  const [vName, vValue] = trimmed.split('=').map(s => s.trim());
                  variables[vName] = evaluateExpression(vValue);
                }
              }
            }
            skipLines = j - i - 1;
          }
        }
      } else if (line.includes('=')) {
        const [varName, value] = line.split('=').map(s => s.trim());
        variables[varName] = evaluateExpression(value);
      } else if (line.startsWith('print(') && line.endsWith(')')) {
        const content = line.slice(6, -1).trim();
        const value = evaluateExpression(content);
        outputLines.push(String(value));
      } else if (line.includes('(') && line.includes(')')) {
        // Function call
        const match = line.match(/(\w+)\((.*?)\)/);
        if (match) {
          const [, funcName, args] = match;
          if (functions[funcName]) {
            const argValues = args.split(',').map(a => evaluateExpression(a.trim()));
            const func = functions[funcName];
            // Simple function execution - for now, just handle print statements
            for (const bodyLine of func.body) {
              if (bodyLine.trim().startsWith('print(')) {
                let printContent = bodyLine.trim().slice(6, -1);
                // Replace parameters
                func.params.forEach((param, idx) => {
                  printContent = printContent.replace(new RegExp(`\\b${param}\\b`, 'g'), String(argValues[idx]));
                });
                const value = evaluateExpression(printContent);
                outputLines.push(String(value));
              }
            }
          }
        }
      }
    }

    return outputLines.join('\n');
  };

  const handleRun = async () => {
    setIsRunning(true);
    // Simulate running
    await new Promise(resolve => setTimeout(resolve, 500));
    const result = mockInterpreter(code);
    setOutput(result);
    setIsRunning(false);
  };

  const checkSolution = () => {
    if (expectedOutput && output.trim() === expectedOutput.trim()) {
      return "Correct!";
    }
    return "Try again.";
  };

  return (
    <div className="my-4 rounded-lg border border-[var(--border-strong)] bg-[var(--surface-1)]">
      <div className="border-b border-[var(--border-strong)] bg-[var(--surface-2)] px-4 py-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-[var(--text-muted)]">
            {language}
          </span>
          <div className="flex gap-2">
            <Button
              onClick={handleRun}
              disabled={isRunning}
              size="sm"
              variant="primary"
            >
              {isRunning ? "Running..." : "Run"}
            </Button>
            {isExercise && solution && (
              <Button
                onClick={() => setShowSolution(!showSolution)}
                size="sm"
                variant="outline"
              >
                {showSolution ? "Hide Solution" : "Show Solution"}
              </Button>
            )}
          </div>
        </div>
      </div>
      <textarea
        value={code}
        onChange={(e) => setCode(e.target.value)}
        className="w-full resize-none border-0 bg-transparent p-4 font-mono text-sm text-[var(--foreground)] focus:outline-none"
        rows={Math.max(code.split('\n').length, 5)}
        spellCheck={false}
      />
      {showOutput && (
        <div className="border-t border-[var(--border-strong)] bg-[var(--surface-2)]">
          <div className="px-4 py-2">
            <span className="text-sm font-medium text-[var(--text-muted)]">
              Output:
            </span>
          </div>
          <pre className="px-4 pb-4 font-mono text-sm text-[var(--foreground)]">
            {output || "Click Run to see output"}
          </pre>
          {isExercise && expectedOutput && (
            <div className="px-4 pb-4">
              <span className="text-sm text-[var(--text-soft)]">
                {checkSolution()}
              </span>
            </div>
          )}
        </div>
      )}
      {showSolution && solution && (
        <div className="border-t border-[var(--border-strong)] bg-green-50 dark:bg-green-900/20">
          <div className="px-4 py-2">
            <span className="text-sm font-medium text-green-700 dark:text-green-300">
              Solution:
            </span>
          </div>
          <pre className="px-4 pb-4 font-mono text-sm text-green-800 dark:text-green-200">
            {solution}
          </pre>
        </div>
      )}
    </div>
  );
}