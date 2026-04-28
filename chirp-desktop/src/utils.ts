export const sampleText =
  "Real change begins when your hope becomes stronger than your excuses, and your actions start matching the person you want to become.";

export const formatBytes = (bytes: number) => {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 MB";
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(bytes > 1024 * 1024 * 1024 ? 2 : 0)} MB`;
};

export const cn = (...classes: Array<string | false | null | undefined>) => classes.filter(Boolean).join(" ");
