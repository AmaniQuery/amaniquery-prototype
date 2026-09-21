"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Code,
  Book,
  Globe,
  Zap,
  Shield,
  Database,
  FileText,
  ExternalLink,
  Copy,
  Check,
  ArrowLeft,
  Terminal,
  Cpu,
  Scale,
  Newspaper,
  CheckCircle2
} from "lucide-react"
import { AnimatedIDE } from "@/components/animated-ide"
import { ChatStreamDemo } from "@/components/chat-stream-demo"
import {
  DEVELOPER_KIT_VERSION,
  LAST_UPDATED,
  MASTER_SYSTEM_PROMPT,
  HYBRID_RAG_PROMPT,
  LEGAL_SPECIALIST_PROMPT,
  NEWS_SPECIALIST_PROMPT,
  PYDANTIC_VALIDATOR,
  GOLDEN_TEST_CASE
} from "./constants"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

function CodeBlock({ code, id, language = "text", copiedSection, onCopy }: { code: string, id: string, language?: string, copiedSection: string | null, onCopy: (text: string, id: string) => void }) {
  return (
    <div className="relative mt-4 group">
      <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <Button
          variant="secondary"
          size="sm"
          className="h-8 w-8 p-0"
          onClick={() => onCopy(code, id)}
        >
          {copiedSection === id ? (
            <Check className="h-4 w-4 text-green-500" />
          ) : (
            <Copy className="h-4 w-4" />
          )}
        </Button>
      </div>
      <pre className="bg-muted/50 p-4 rounded-lg overflow-x-auto border border-border/50 text-sm font-mono leading-relaxed">
        <code className={`language-${language}`}>{code}</code>
      </pre>
    </div>
  )
}

export default function DevelopersPage() {
  const [copiedSection, setCopiedSection] = useState<string | null>(null)

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedSection(id)
    setTimeout(() => setCopiedSection(null), 2000)
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link href="/">
            <Button variant="ghost" className="mb-4 pl-0 hover:bg-transparent hover:text-primary">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Home
            </Button>
          </Link>
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-4xl font-bold tracking-tight">Developer Hub</h1>
                <Badge variant="secondary" className="text-sm">v{DEVELOPER_KIT_VERSION}</Badge>
              </div>
              <p className="text-muted-foreground text-lg max-w-2xl">
                Build compliant legal AI applications with the AmaniQuery Developer Kit.
                Access system prompts, validators, and testing resources.
              </p>
            </div>
            <div className="flex gap-2">
              <a href={`${API_BASE_URL}/docs`} target="_blank" rel="noopener noreferrer">
                <Button variant="outline">
                  <ExternalLink className="w-4 h-4 mr-2" />
                  API Reference
                </Button>
              </a>
            </div>
          </div>
        </div>

        {/* Interactive Demos */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
          <AnimatedIDE />
          <ChatStreamDemo />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-9 space-y-8">
            
            <Tabs defaultValue="prompts" className="w-full">
              <TabsList className="w-full justify-start h-auto p-1 bg-muted/50 mb-6 overflow-x-auto flex-nowrap">
                <TabsTrigger value="prompts" className="flex items-center gap-2 px-4 py-2">
                  <Terminal className="w-4 h-4" />
                  System Prompts
                </TabsTrigger>
                <TabsTrigger value="specialists" className="flex items-center gap-2 px-4 py-2">
                  <Cpu className="w-4 h-4" />
                  Specialist Agents
                </TabsTrigger>
                <TabsTrigger value="validators" className="flex items-center gap-2 px-4 py-2">
                  <Shield className="w-4 h-4" />
                  Validators
                </TabsTrigger>
                <TabsTrigger value="testing" className="flex items-center gap-2 px-4 py-2">
                  <CheckCircle2 className="w-4 h-4" />
                  Testing
                </TabsTrigger>
              </TabsList>

              {/* System Prompts Tab */}
              <TabsContent value="prompts" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Zap className="w-5 h-5 text-yellow-500" />
                      Master System Prompt
                    </CardTitle>
                    <CardDescription>
                      The core instruction set for the AmaniQuery engine. Handles intent detection, 
                      language switching (English/Swahili), and response formatting.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <CodeBlock code={MASTER_SYSTEM_PROMPT} id="master-prompt" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Database className="w-5 h-5 text-blue-500" />
                      Hybrid RAG Mode
                    </CardTitle>
                    <CardDescription>
                      Specialized prompt for retrieving and synthesizing legal sources. 
                      Enforces citation rules and structured analysis.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <CodeBlock code={HYBRID_RAG_PROMPT} id="rag-prompt" />
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Specialist Agents Tab */}
              <TabsContent value="specialists" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Scale className="w-5 h-5 text-indigo-500" />
                      Legal Content Specialist
                    </CardTitle>
                    <CardDescription>
                      Generates court-ready legal analysis with Bluebook citations, 
                      IRAC structure, and statutory comparisons.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <CodeBlock code={LEGAL_SPECIALIST_PROMPT} id="legal-prompt" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Newspaper className="w-5 h-5 text-rose-500" />
                      News & Parliamentary Specialist
                    </CardTitle>
                    <CardDescription>
                      Formats Hansard records, voting outcomes, and news summaries 
                      with journalistic precision and attribution.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <CodeBlock code={NEWS_SPECIALIST_PROMPT} id="news-prompt" />
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Validators Tab */}
              <TabsContent value="validators" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="w-5 h-5 text-green-500" />
                      Pydantic Response Validator
                    </CardTitle>
                    <CardDescription>
                      Python code to validate the structural integrity of agent responses, 
                      ensuring no reasoning leakage and proper citation format.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <CodeBlock code={PYDANTIC_VALIDATOR} id="pydantic-validator" language="python" />
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Testing Tab */}
              <TabsContent value="testing" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <CheckCircle2 className="w-5 h-5 text-teal-500" />
                      Golden Test Cases
                    </CardTitle>
                    <CardDescription>
                      Standardized Q&A pairs for regression testing the agent&apos;s performance.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <CodeBlock code={GOLDEN_TEST_CASE} id="test-case" />
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-3 space-y-6">
            {/* Resources */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center text-lg">
                  <Book className="w-5 h-5 mr-2" />
                  Resources
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <a
                  href={`${API_BASE_URL}/docs`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center text-sm text-muted-foreground hover:text-primary transition-colors"
                >
                  <FileText className="w-4 h-4 mr-2" />
                  Interactive API Docs
                </a>
                <a
                  href="https://github.com/Benaah/amaniquery"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center text-sm text-muted-foreground hover:text-primary transition-colors"
                >
                  <Code className="w-4 h-4 mr-2" />
                  GitHub Repository
                </a>
                <Link href="/blog" className="flex items-center text-sm text-muted-foreground hover:text-primary transition-colors">
                  <Globe className="w-4 h-4 mr-2" />
                  Engineering Blog
                </Link>
              </CardContent>
            </Card>

            {/* Kit Info */}
            <Card className="bg-primary/5 border-primary/20">
              <CardHeader>
                <CardTitle className="flex items-center text-lg">
                  <Database className="w-5 h-5 mr-2" />
                  Kit Details
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex justify-between py-1 border-b border-border/50">
                  <span className="text-muted-foreground">Version</span>
                  <span className="font-mono font-medium">{DEVELOPER_KIT_VERSION}</span>
                </div>
                <div className="flex justify-between py-1 border-b border-border/50">
                  <span className="text-muted-foreground">Updated</span>
                  <span className="font-mono font-medium">{LAST_UPDATED}</span>
                </div>
                <div className="flex justify-between py-1">
                  <span className="text-muted-foreground">Status</span>
                  <Badge variant="outline" className="text-xs bg-green-500/10 text-green-600 border-green-500/20">
                    Production
                  </Badge>
                </div>
              </CardContent>
            </Card>

            {/* Support */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center text-lg">
                  <Globe className="w-5 h-5 mr-2" />
                  Support
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <a
                  href="https://github.com/Benaah/amaniquery/issues"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center text-sm text-muted-foreground hover:text-primary transition-colors"
                >
                  Report Issues
                </a>
                <a
                  href="https://github.com/Benaah/amaniquery/discussions"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center text-sm text-muted-foreground hover:text-primary transition-colors"
                >
                  Community Discussions
                </a>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
