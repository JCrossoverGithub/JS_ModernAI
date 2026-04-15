import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useState } from "react";
import { Copy, Check } from "lucide-react";

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      onClick={() => {
        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      }}
      className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white transition-colors"
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export default function MarkdownRenderer({ content }: { content: string }) {
  return (
    <div className="prose-chat">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ className, children, ...rest }) {
            const lang = className?.replace("language-", "") || "";
            const code = String(children).replace(/\n$/, "");
            const isBlock = Boolean(className);

            if (isBlock) {
              return (
                <div className="relative group my-3">
                  <div className="flex items-center justify-between bg-white/[0.03] border border-white/[0.06] rounded-t-[10px] px-4 py-2">
                    <span className="text-xs text-slate-500 font-medium">
                      {lang || "code"}
                    </span>
                    <CopyButton text={code} />
                  </div>
                  <pre className="!mt-0 !rounded-t-none !border-t-0">
                    <code className={className}>{code}</code>
                  </pre>
                </div>
              );
            }

            return (
              <code className={className} {...rest}>
                {children}
              </code>
            );
          },
          pre({ children }) {
            return <>{children}</>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
