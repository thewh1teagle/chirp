export type ModelBundle = {
  installed: boolean;
  model_path: string;
  codec_path: string;
  model_dir: string;
  version: string;
  url: string;
};

export type RunnerInfo = {
  base_url: string;
};

export type DownloadProgress = {
  downloaded: number;
  total?: number | null;
  progress?: number | null;
  stage: "downloading" | "extracting";
};

export type CreateStep = "idle" | "starting" | "loading" | "creating" | "done";
