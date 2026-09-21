"use client"

import { useState, useRef, useEffect } from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { 
  Bot, 
  User, 
  Copy, 
  RotateCw, 
  ThumbsUp, 
  ThumbsDown,
  ExternalLink,
  Check,
  Pencil,
  AlertTriangle,
  RefreshCw,
  Square,
  FileText,
  FileDown
} from "lucide-react"
import { Textarea } from "@/components/ui/textarea"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import rehypeHighlight from "rehype-highlight"
import type { Message, Source } from "./types"

interface AmaniMessageProps {
  message: Message
  onCopy: (content: string) => void
  onRegenerate: (messageId: string) => void
  onFeedback: (messageId: string, type: "like" | "dislike") => void
  isLoading?: boolean
  showFeedback?: boolean
  isEditing?: boolean
  editingContent?: string
  onEditChange?: (content: string) => void
  onSaveEdit?: (messageId: string) => void
  onCancelEdit?: () => void
  onStartEdit?: (message: Message) => void
  onCopyFailed?: (message: Message) => void
  onEditFailed?: (message: Message) => void
  onResendFailed?: (message: Message) => void
}

interface InlineCitationProps {
  source: Source
  index: number
  onHover: (source: Source | null, event: React.MouseEvent) => void
}

function InlineCitation({ source, index, onHover }: InlineCitationProps) {
  return (
    <button
      type="button"
      className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 text-[10px] font-medium text-primary bg-primary/10 rounded-full hover:bg-primary hover:text-primary-foreground transition-all ml-0.5 align-top mt-0.5"
      onMouseEnter={(e) => onHover(source, e)}
      onMouseLeave={(e) => onHover(null, e)}
      onClick={(e) => {
        e.stopPropagation()
        window.open(source.url, '_blank', 'noopener,noreferrer')
      }}
    >
      {index}
    </button>
  )
}

interface CitationTooltipProps {
  source: Source | null
  position: { x: number; y: number }
}

function CitationTooltip({ source, position }: CitationTooltipProps) {
  if (!source) return null

  return (
    <div 
      className="fixed z-[100] bg-popover text-popover-foreground border rounded-lg shadow-xl p-3 max-w-[300px] animate-in fade-in zoom-in-95 duration-100"
      style={{ left: position.x, top: position.y }}
    >
      <div className="space-y-1.5">
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{source.source_name}</span>
          <ExternalLink className="w-3 h-3 text-muted-foreground" />
        </div>
        <h4 className="text-sm font-medium leading-snug">{source.title}</h4>
        {source.excerpt && (
          <p className="text-xs text-muted-foreground line-clamp-3 leading-relaxed bg-muted/50 p-2 rounded-md">
            {source.excerpt}
          </p>
        )}
      </div>
    </div>
  )
}

function processContentWithCitations(content: string, sources?: Source[]) {
  if (!sources || sources.length === 0) return content

  const processedContent = content
  // Regex to match [1], [2], etc.
  const citationPattern = /\[(\d+)\]/g
  
  // We don't replace here anymore, we let the markdown parser handle the text, 
  // but we might want to inject components if we wrote a custom remark plugin.
  // For simplicity with ReactMarkdown, we'll do a string replacement to a custom syntax 
  // or just rely on the fact that we can parse these if we wanted to.
  // BUT, to render our custom component, replacing with a unique string we can split on is easier.
  
  // Actually, a safer way is to replace [n] with a special marker that we can split by later.
  return processedContent.replace(citationPattern, (match, num) => {
    const index = parseInt(num) - 1
    if (sources[index]) {
      return `{{CITATION:${index}}}`
    }
    return match
  })
}

function MarkdownCodeBlock({ children, className, ...props }: { children: React.ReactNode, className?: string }) {
  const [copied, setCopied] = useState(false)
  const match = /language-(\w+)/.exec(className || "")
  const isInline = !match && !String(children).includes("\n")

  const handleCopyCode = async () => {
    await navigator.clipboard.writeText(String(children).replace(/\n$/, ''))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (isInline) {
    return (
      <code className="bg-muted px-1.5 py-0.5 rounded text-sm font-mono text-foreground border border-border" {...props}>
        {children}
      </code>
    )
  }

  return (
    <div className="relative my-4 rounded-lg border border-border bg-muted/50 overflow-hidden">
      <div className="flex items-center justify-between px-3 py-1.5 bg-muted/80 border-b border-border text-xs text-muted-foreground font-mono">
        <span>{match?.[1] || 'code'}</span>
        <button
          type="button"
          onClick={handleCopyCode}
          className="flex items-center gap-1 hover:text-foreground transition-colors"
        >
          {copied ? (
            <><Check className="w-3 h-3 text-green-500" /> Copied</>
          ) : (
            <><Copy className="w-3 h-3" /> Copy</>
          )}
        </button>
      </div>
      <div className="overflow-x-auto p-4">
        <code className={cn("text-sm font-mono block", className)} {...props}>
          {children}
        </code>
      </div>
    </div>
  )
}

export function AmaniMessage({ 
  message, 
  onCopy, 
  onRegenerate, 
  onFeedback, 
  isLoading = false,
  showFeedback = true,
  isEditing = false,
  editingContent = "",
  onEditChange,
  onSaveEdit,
  onCancelEdit,
  onStartEdit,
  onCopyFailed,
  onEditFailed,
  onResendFailed
}: AmaniMessageProps) {
  const [hoveredSource, setHoveredSource] = useState<Source | null>(null)
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 })
  const [hasCopied, setHasCopied] = useState(false)
  const [showActions, setShowActions] = useState(false)
  const messageRef = useRef<HTMLDivElement>(null)

  const isUser = message.role === "user"

  const handleCopy = async () => {
    await onCopy(message.content)
    setHasCopied(true)
    setTimeout(() => setHasCopied(false), 2000)
  }

  const handleSourceHover = (source: Source | null, event?: React.MouseEvent) => {
    setHoveredSource(source)
    if (source && event) {
      const rect = event.currentTarget.getBoundingClientRect()
      // Calculate position to keep it on screen
      const x = Math.min(rect.right + 10, window.innerWidth - 320)
      const y = Math.min(rect.top, window.innerHeight - 200)
      setTooltipPosition({ x, y })
    }
  }

  const renderContent = () => {
    const processedContent = processContentWithCitations(message.content, message.sources)
    // Split by our custom marker
    const parts = processedContent.split(/(\{\{CITATION:\d+\}\})/g)

    return (
      <div className="space-y-4">
        {/* Failed Message State */}
        {message.failed && (
           <div className="flex items-center gap-3 p-3 border border-destructive/20 bg-destructive/5 rounded-lg text-destructive text-sm">
                <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                <span className="flex-1">Message failed to send</span>
                <div className="flex items-center gap-1">
                    {onResendFailed && (
                        <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 hover:bg-destructive/10 hover:text-destructive"
                            onClick={() => onResendFailed(message)}
                        >
                            Retry
                        </Button>
                    )}
                     {onEditFailed && (
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-7 w-7 hover:bg-destructive/10 hover:text-destructive"
                            onClick={() => onEditFailed(message)}
                        >
                            <Pencil className="w-3.5 h-3.5" />
                        </Button>
                    )}
                </div>
            </div>
        )}

        {/* Stopped / Interrupted Message State */}
        {message.stopped && (
          <div className="flex items-center gap-2 mb-3 text-xs text-muted-foreground">
            <Square className="w-3 h-3 fill-muted-foreground" />
            <span>Generation stopped</span>
          </div>
        )}

        {!message.failed && (
          <div className="prose prose-sm md:prose-base dark:prose-invert max-w-none prose-p:leading-relaxed prose-pre:p-0 prose-pre:bg-transparent">
            {parts.map((part, i) => {
              if (part.startsWith("{{CITATION:")) {
                const index = parseInt(part.replace("{{CITATION:", "").replace("}}", ""))
                const source = message.sources?.[index]
                if (source) {
                  return (
                    <InlineCitation
                      key={i}
                      source={source}
                      index={index + 1}
                      onHover={handleSourceHover}
                    />
                  )
                }
                return null
              }
              
              return (
                <ReactMarkdown
                  key={i}
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                  components={{
                    p: ({ children }) => <span className="block mb-4 last:mb-0 leading-7">{children}</span>,
                    h1: ({ children }) => <h1 className="text-lg font-bold mb-3 mt-6">{children}</h1>,
                    h2: ({ children }) => <h2 className="text-base font-bold mb-2 mt-4">{children}</h2>,
                    h3: ({ children }) => <h3 className="text-sm font-bold mb-2 mt-4 uppercase tracking-wide opacity-80">{children}</h3>,
                    ul: ({ children }) => <ul className="list-disc list-outside ml-4 mb-4 space-y-1">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal list-outside ml-4 mb-4 space-y-1">{children}</ol>,
                    li: ({ children }) => <li className="pl-1 mb-1">{children}</li>,
                    blockquote: ({ children }) => (
                      <blockquote className="border-l-2 border-primary/30 pl-4 my-4 italic text-muted-foreground bg-muted/30 py-2 pr-2 rounded-r">
                        {children}
                      </blockquote>
                    ),
                    table: ({ children }) => (
                      <div className="overflow-x-auto my-4 border border-border rounded-lg">
                        <table className="min-w-full divide-y divide-border text-sm">
                          {children}
                        </table>
                      </div>
                    ),
                    thead: ({ children }) => (
                      <thead className="bg-muted/80">{children}</thead>
                    ),
                    th: ({ children }) => (
                      <th className="px-4 py-2.5 text-left font-semibold text-foreground whitespace-nowrap">{children}</th>
                    ),
                    td: ({ children }) => (
                      <td className="px-4 py-2 border-t border-border whitespace-nowrap">{children}</td>
                    ),
                    tr: ({ children }) => (
                      <tr className="even:bg-muted/30">{children}</tr>
                    ),
                    img: ({ src, alt }) => (
                      <img src={src} alt={alt || ""} className="max-w-full h-auto rounded-lg border border-border my-4" loading="lazy" />
                    ),
                    code: MarkdownCodeBlock,
                    input: ({ type, checked, ...props }) => {
                      if (type === "checkbox") {
                        return (
                          <input
                            type="checkbox"
                            checked={checked}
                            readOnly
                            className="accent-primary w-4 h-4 rounded border-border mr-2 -ml-5 mt-1"
                            {...props}
                          />
                        )
                      }
                      return <input {...props} />
                    },
                    hr: () => <hr className="my-6 border-border" />,
                    a: ({ children, href }) => (
                      <a href={href} target="_blank" rel="noopener noreferrer" className="text-primary underline underline-offset-4 hover:text-primary/80 transition-colors">
                        {children}
                      </a>
                    ),
                  }}
                >
                  {part}
                </ReactMarkdown>
              )
            })}
          </div>
        )}

        {message.research_bundle && (
          <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t border-border/50">
            {message.research_bundle.has_pdf && message.research_bundle.download_urls.pdf && (
              <a
                href={message.research_bundle.download_urls.pdf}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
              >
                <FileText className="w-3.5 h-3.5" />
                Download PDF
              </a>
            )}
            {message.research_bundle.has_docx && message.research_bundle.download_urls.docx && (
              <a
                href={message.research_bundle.download_urls.docx}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
              >
                <FileDown className="w-3.5 h-3.5" />
                Download DOCX
              </a>
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div 
      ref={messageRef}
      className={cn(
        "group relative flex gap-4 p-0 rounded-2xl transition-all duration-200",
        isUser ? "flex-row-reverse" : "flex-row",
        isLoading && "opacity-90"
      )}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => {
        setShowActions(false)
        setHoveredSource(null)
      }}
    >
      {/* Avatar */}
      <div className={cn(
        "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center border shadow-sm mt-1",
        isUser ? "bg-secondary text-foreground border-border" : "bg-primary text-primary-foreground border-transparent"
      )}>
        {isUser ? (
          <User className="w-4 h-4" />
        ) : (
          <Bot className="w-4 h-4" />
        )}
      </div>

      {/* Message Content */}
      <div className={cn(
        "flex-1 max-w-3xl space-y-2",
        isUser ? "text-right" : "text-left"
      )}>
        <div className={cn(
          "inline-block text-base leading-relaxed max-w-full",
          isUser 
            ? "bg-secondary text-secondary-foreground rounded-2xl rounded-tr-sm px-5 py-3" 
            : "bg-transparent text-foreground p-0" 
        )}>
          {isEditing ? (
            <div className="flex flex-col gap-3 min-w-[300px] w-full bg-background border rounded-xl p-3 shadow-sm text-left">
                <Textarea
                    value={editingContent}
                    onChange={(e) => onEditChange?.(e.target.value)}
                    className="bg-transparent border-input focus:border-primary resize-none min-h-[100px]"
                />
                <div className="flex justify-end gap-2">
                    <Button
                        size="sm"
                        variant="ghost"
                        onClick={onCancelEdit}
                    >
                        Cancel
                    </Button>
                    <Button
                        size="sm"
                        onClick={() => onSaveEdit?.(message.id)}
                    >
                        Save
                    </Button>
                </div>
            </div>
          ) : (
            renderContent()
          )}
        </div>

        {/* Message Actions */}
        {showActions && !isEditing && !message.failed && (
            <div className={cn(
                "flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200",
                isUser ? "justify-end pr-1" : "justify-start pl-0"
            )}>
          
          {!isUser && (
            <>
             <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-muted-foreground hover:text-foreground hover:bg-secondary rounded-md"
              onClick={handleCopy}
              title="Copy"
            >
              {hasCopied ? <Check className="w-3.5 h-3.5 text-green-500" /> : <Copy className="w-3.5 h-3.5" />}
            </Button>

            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-muted-foreground hover:text-foreground hover:bg-secondary rounded-md"
              onClick={() => onRegenerate(message.id)}
              title="Regenerate"
            >
              <RotateCw className="w-3.5 h-3.5" />
            </Button>
            
             {showFeedback && (
              <>
                <Button
                  variant="ghost"
                  size="icon"
                  className={cn(
                    "h-6 w-6 rounded-md hover:bg-secondary",
                    message.feedback_type === "like" 
                      ? "text-green-600" 
                      : "text-muted-foreground hover:text-green-600"
                  )}
                  onClick={() => onFeedback(message.id, "like")}
                >
                  <ThumbsUp className="w-3.5 h-3.5" />
                </Button>

                <Button
                  variant="ghost"
                  size="icon"
                  className={cn(
                    "h-6 w-6 rounded-md hover:bg-secondary",
                    message.feedback_type === "dislike" 
                      ? "text-red-600" 
                      : "text-muted-foreground hover:text-red-600"
                  )}
                  onClick={() => onFeedback(message.id, "dislike")}
                >
                  <ThumbsDown className="w-3.5 h-3.5" />
                </Button>
              </>
            )}
            </>
          )}

           {isUser && onStartEdit && (
                <div className="flex items-center gap-1">
                    <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 text-muted-foreground hover:text-foreground hover:bg-background/20 rounded-md"
                        onClick={() => onStartEdit(message)}
                    >
                        <Pencil className="w-3.5 h-3.5" />
                    </Button>
                </div>
           )}
          </div>
        )}
      </div>

      {/* Citation Tooltip */}
      <CitationTooltip source={hoveredSource} position={tooltipPosition} />
    </div>
  )
}