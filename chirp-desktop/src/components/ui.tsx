import { motion } from "framer-motion";
import { X } from "lucide-react";
import { ComponentProps, ReactNode } from "react";
import { cn } from "../utils";

type ButtonVariant = "primary" | "secondary" | "ghost" | "outline";

export function Button({ className, variant = "primary", ...props }: ComponentProps<"button"> & { variant?: ButtonVariant }) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-lg text-sm font-semibold cursor-pointer transition-all active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary" && "bg-[#111111] text-white hover:bg-black shadow-sm",
        variant === "secondary" && "bg-white border border-border/60 text-primary hover:bg-background hover:border-border shadow-sm",
        variant === "outline" && "border border-border/80 bg-white text-primary hover:border-primary hover:bg-white shadow-sm",
        variant === "ghost" && "bg-transparent text-secondary hover:bg-background hover:text-primary",
        className,
      )}
      {...props}
    />
  );
}

export function Card({ className, ...props }: ComponentProps<"div">) {
  return <div className={cn("rounded-xl border border-border/60 bg-surface shadow-sm", className)} {...props} />;
}

export function Progress({ value }: { value: number }) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-border/30">
      <motion.div
        className="h-full bg-primary"
        initial={{ width: 0 }}
        animate={{ width: `${Math.max(0, Math.min(100, value))}%` }}
        transition={{ type: "spring", bounce: 0, duration: 0.5 }}
      />
    </div>
  );
}

export function Brand({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "inline-grid h-8 min-w-[64px] place-items-center rounded-lg border-2 border-primary bg-primary px-3 text-[10px] font-black uppercase tracking-[0.2em] text-white",
        className,
      )}
    >
      Chirp
    </div>
  );
}

export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn("mb-2 block text-[10px] font-bold uppercase tracking-[0.15em] text-secondary opacity-60", className)}>{children}</span>;
}

export function ErrorBlock({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("flex gap-3 rounded-xl border border-red-100 bg-red-50/50 p-4 text-sm text-red-800", className)}>
      <X className="h-4 w-4 shrink-0" />
      <pre className="overflow-auto whitespace-pre-wrap break-words font-medium">{children}</pre>
    </div>
  );
}
